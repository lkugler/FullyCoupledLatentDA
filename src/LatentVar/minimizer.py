
import time
from time import sleep
import torch
from autoencoder_da.utilities import standardise_destandardise
from LatentVar.interpolator import interpolate_irregular_grid

# ------------------------------------------------------
# 3D-Var
# ------------------------------------------------------

def latent3DVar_algorithm_preconditioned(
        input_dict,  # dictionary containing all the inputs (latent_bg, obs_vec, B_matrix_sqrt, obs_lats, obs_lons, obs_qty_idx, R_matrix_inv)
        device: str,  # device to run the algorithm on
        init_lr: float=0.3,   # float
        max_num_steps=100,
        factor_lr=0.5,  # Factor by which the learning rate is reduced
        rtol_stop=0.01,   # Relative tolerance for convergence criterion
        minimum_lr=1e-4
):
    
    latent_bg = input_dict['latent_bg']    # torch tensor, shape (nnode, latent_dim)
    obs_vec = input_dict['obs_vec']        # torch tensor, shape (nobs, 1)
    B_matrix_sqrt = input_dict['B_matrix_sqrt']    # torch tensor, shape (nnode, latent_dim), elements correspond to full B diagonal
    obs_lats = input_dict['obs_lats']  # torch tensor, shape (nobs,)
    obs_lons = input_dict['obs_lons']  # torch tensor, shape (nobs,)
    obs_qty_idx = input_dict['obs_qty_idx']    # list with observed qty indices - used for interpolation in H operator
    R_matrix_inv = input_dict['R_matrix_inv']  # torch tensor, shape (nobs, 1)
    grid_lats = input_dict['grid_lats']  # torch tensor, shape (nnode,)
    grid_lons = input_dict['grid_lons']  # torch tensor, shape (nnode,)
    AE_model = input_dict['AE_model']
    AE_props = input_dict['AE_props']
    AE_scalers_mean = input_dict['AE_scalers_mean']
    AE_scalers_std = input_dict['AE_scalers_std']
        
    
    '''All torch tensors are float32'''
    # Though most of torch tensors are already on GPU, we resend them (for those that are already there this happens instantaneously)
    latent_bg = latent_bg.to(device)
    obs_vec = obs_vec.to(device)
    B_matrix_sqrt = B_matrix_sqrt.to(device)
    obs_lats = obs_lats.to(device)
    obs_lons = obs_lons.to(device)
    R_matrix_inv = R_matrix_inv.to(device)

    corrected_chi = torch.zeros(latent_bg.shape).to(device) # = B_matrix_sqrt_inv @ (latent_bg - latent_bg).clone()
    corrected_chi.requires_grad = True


    # ------------------------------------------------------
    # Prepare minimisation trackers and settings
    # ------------------------------------------------------

    # Setting the optimizer for stochastic gradient descend during 3D-Var
    optimizer = torch.optim.SGD([corrected_chi], lr=init_lr)
    # Variable to keep track of the best loss
    best_J = float('inf')
    # Minimisation step with the best loss
    best_J_step = 0
    # Latent state at the best step
    best_latent_vec = latent_bg.clone().detach() # = (B_matrix_sqrt @ corrected_chi_vec.clone().detach() + latent_bg)


    all_J = [] # Store the values of cost function at each minimisation step
    all_Jb = [] # Store the values of background term at each minimisation step
    all_Jo = [] # Store the values of observation term of cost function at each minimisation step
    all_grad_J = [] # Store the values of the cost function gradient's Euclidean norm at each minimisation step

    # Ending step of the minimisation process (retains this value if the convergence is not reached before that)
    ending_step = max_num_steps

    # ------------------------------------------------------
    # Preconditioned 3D-VAR cost function
    # ------------------------------------------------------
    def latent3DVar_cost():
        # ------------------------------------------------------
        # Background term
        # ------------------------------------------------------

        # If chi was a vector with shape (nnode*latent_dim, 1): J_b = 1/2 * chi**T @ chi
        # The formulation below is equivalent, faster, and works also for chi with shape (nnode, latent_dim)
        J_b = 1 /2 * torch.sum(torch.sum(corrected_chi * corrected_chi, axis=1))

        # ------------------------------------------------------
        # Observation term
        # ------------------------------------------------------
        # J_o = 1/2 * (y - H(S**-1(D(z))))**T R**-1 (y - H(S**-1(D(z))))

        # 1) S**-1(D(z)), z = zb + B_sqrt * chi
        decoded_lat_vec_dest = standardise_destandardise(
            input_fields = AE_model.decode(latent_bg + B_matrix_sqrt * corrected_chi),
            AE_props = AE_props,
            scalers_mean = AE_scalers_mean,
            scalers_std = AE_scalers_std,
            action = 'destandardise',
            device = device
        )

        # 2) Apply observation operator H
        # TODO: Here, HSD is not used as combined operator, compared to the incremental version
        HSDz = interpolate_irregular_grid(grid_lats, grid_lons, decoded_lat_vec_dest[:, obs_qty_idx], obs_lats, obs_lons).T
        # print('HSDz', HSDz)
        # input('....')

        # 3) Compute J_o with a similar speedup as J_b
        J_o = 1 / 2 * torch.sum(torch.sum((obs_vec - HSDz) * R_matrix_inv * (obs_vec - HSDz), axis=1))


        # ------------------------------------------------------
        # Compute sum of terms and return them
        # ------------------------------------------------------

        J = torch.add(J_b, J_o)

        return J, J_b, J_o


    # ------------------------------------------------------
    # Minimisation algorithm
    # ------------------------------------------------------
    for step in range(1, max_num_steps + 1):
        J, Jb, Jo = latent3DVar_cost()  # Get current values of the cost function

        all_J.append(J.item())  # Store current value of the cost function
        all_Jb.append(Jb.item())  # Store current value of the background term
        all_Jo.append(Jo.item())  # Store current value of the observation term
        previous_chi = corrected_chi.clone().detach()  # Store current latent vector
        J.backward(retain_graph=True)  # Compute current gradient
        grad_J = corrected_chi.grad  # Get the gradient of the latent vector
        norm_grad_J = torch.norm(grad_J, p=2)
        all_grad_J.append(norm_grad_J)  # Store the Euclidean norm of current gradient

        optimizer.step()  # Change the latent vector according to its current gradient
        optimizer.zero_grad()  # Clear the gradients for the next iteration

        # Check for improvement in loss (for learning rate)
        if J < best_J:
            best_J = J
            best_grad_J = norm_grad_J
            best_J_step = step
            best_latent_vec = (B_matrix_sqrt * previous_chi + latent_bg)
        else:
            current_lr = optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * factor_lr, minimum_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Check for improvement in gradient (for stopping criterion)
        if step >= 2:
            if all_grad_J[-1] / all_grad_J[0] < rtol_stop:
                ending_step = step
                break

        # Monitor the minimisation procedure
        # We print ensemble member index, minimisation step, cost function value,
        # the ratio between the cost function in this and the previous step, number of steps after the last update of the best latent vector
        if step == 1:
            print(f'\nInitial J {J.item():4f}, initial grad J {all_grad_J[0].item():4f}')
        if step > 1:
            print(f'Step {step}, J {J.item():4f}, ratio {all_J[-1] / all_J[-2]:4f},',
                  f'grad J {all_grad_J[-1].item():4f}, ratio {(all_grad_J[-1] / all_grad_J[0]).item():4f}')

    ending_J = J.item()
    print(f'Ending step {ending_step}, ending J {J.item():4f}, best step {best_J_step},',
          f'best J {best_J.item():4f}, best grad J ratio {(best_grad_J / all_grad_J[0]).item():4f}')
    
    if ending_J > all_J[0]:
        raise RuntimeError('Ending J is worse than initial J!')  # lkugler

    # This kind of output may be a bit clumsy, however, it has to be done this way in case of parallelization,
    # so we decided to do it the same way here for the sake of universality.
    return {'out_latent': best_latent_vec, 'best_J': best_J, 'all_J': all_J,
            'all_Jo': all_Jo, 'all_Jb': all_Jb, 'all_grad_J': all_grad_J}

def incremental_latent3DVar(
        input_dict,  # dictionary containing all the inputs (latent_bg, obs_vec, B_matrix_sqrt, obs_lats, obs_lons, obs_qty_idx, R_matrix_inv)
        device: str,  # device to run the algorithm on
        init_lr: float=0.3,   # initial learning rate for the inner loop SGD
        max_outer_loops=4,        # number of outer loop linearizations
        max_inner_loops=20,       # number of inner loop SGD steps
        factor_lr=0.5,
        rtol_stop=0.01,
        minimum_lr=1e-5
):
    latent_bg = input_dict['latent_bg']    # torch tensor, shape (nnode, latent_dim)
    obs_vec = input_dict['obs_vec']        # torch tensor, shape (nobs, 1)
    B_matrix_sqrt = input_dict['B_matrix_sqrt']    # torch tensor, shape (nnode, latent_dim), elements correspond to full B diagonal
    obs_lats = input_dict['obs_lats']  # torch tensor, shape (nobs,)
    obs_lons = input_dict['obs_lons']  # torch tensor, shape (nobs,)
    obs_qty_idx = input_dict['obs_qty_idx']    # list with observed qty indices - used for interpolation in H operator
    R_matrix_inv = input_dict['R_matrix_inv']  # torch tensor, shape (nobs, 1)
    grid_lats = input_dict['grid_lats']  # torch tensor, shape (nnode,)
    grid_lons = input_dict['grid_lons']  # torch tensor, shape (nnode,)
    AE_model = input_dict['AE_model']
    AE_props = input_dict['AE_props']
    AE_scalers_mean = input_dict['AE_scalers_mean']
    AE_scalers_std = input_dict['AE_scalers_std']
    
    

    sleepseconds = 0
    # print('Entered main function')
    # sleep(3)
    latent_bg = latent_bg.to(device)
    obs_vec = obs_vec.to(device)
    B_matrix_sqrt = B_matrix_sqrt.to(device)
    obs_lats = obs_lats.to(device)
    obs_lons = obs_lons.to(device)
    R_matrix_inv = R_matrix_inv.to(device)
    print('R_matrix_inv', R_matrix_inv.size())
    # print('Put stuff to device')
    # sleep(3)

    

    # Decode latent state into physical fields and destandardise
    def full_decoder(latent):
        return standardise_destandardise(
                            input_fields = AE_model.decode(latent),
                            AE_props = AE_props,
                            scalers_mean = AE_scalers_mean,
                            scalers_std = AE_scalers_std,
                            action = 'destandardise',
                            device = device
                        )
    # Observation operator: interpolate model field to obs. locs
    def H_operator(state_field):
        return interpolate_irregular_grid(
                            grid_lats, grid_lons,
                            state_field[:, obs_qty_idx],
                            obs_lats, obs_lons
                        ).T

    # Combined decoder + obs operator
    def HS_decoder(latent):
        return H_operator(full_decoder(latent))

    # Initial latent estimate
    best_latent_vec_overall = latent_bg.clone().detach()
    # current_best_guess_latent = latent_bg.clone().detach()
    
    # Trackers for outer loops
    all_J_outer = []
    all_Jb_outer = []
    all_Jo_outer = []
    all_grad_outer = []

    all_J = []
    all_Jb = []
    all_Jo = []
    all_grad = []

    latents = []

    #initialisation:
    corrected_chi_g = torch.zeros_like(latent_bg).to(device)
    min_lr_counter = 0  # Track how many times the minimum LR has been used
    # print('before entering the loop')
    # sleep(3)
    for outer_step in range(1, max_outer_loops + 1):
        print(f"\n=== Outer loop {outer_step} ===")
        start_time = time.time()  
        # Initialize inner loop increment
        # corrected_chi = 1e-5 * torch.randn_like(latent_bg).to(device)
        corrected_chi = torch.zeros_like(latent_bg).to(device)
        corrected_chi.requires_grad = True

        # sleep(3)
        print('Computing jacobian')
        sleep(sleepseconds)
        J_HSD = torch.autograd.functional.jacobian(
            HS_decoder,
            (latent_bg + B_matrix_sqrt * corrected_chi_g).detach(), 
            create_graph=False
        )  # shape (n_obs, n_state)
        J_HSD = J_HSD.detach()
        # print(torch.isnan(J_HSD).sum())
        # print([i.size() for i in J_HSD])
        # print('Did it')
        # print(corrected_chi.size())

        # just_SD = torch.autograd.functional.jacobian(
        #     full_decoder,
        #     (latent_bg + B_matrix_sqrt * corrected_chi_g).detach(), 
        #     create_graph=False
        # )

        # input('f')

        # Innovation vector at best current latent guess
        H_xb = HS_decoder(latent_bg + B_matrix_sqrt * corrected_chi_g)
        innovation_vec = (obs_vec - H_xb).detach()

        # print('Inovation vector computed')


        # Setting the optimizer for stochastic gradient descend during 3D-Var
        optimizer = torch.optim.SGD([corrected_chi], lr=init_lr) #It will look at the .grad attribute of corrected_chi after you call J.backward(), and then update its value using the update rule
        # Trackers for inner loop
        all_J_inner, all_Jb_inner, all_Jo_inner, all_grad_inner = [], [], [], []
        best_J_inner = float('inf')
        # Detach the outer latent to avoid backward graph errors
        outer_latent_detached = best_latent_vec_overall.detach()
        best_latent_inner = outer_latent_detached.clone() # = (B_matrix_sqrt @ corrected_chi_vec.clone().detach() + latent_bg)
        best_J_inner_step = 0

        for inner_step in range(1, max_inner_loops + 1):
            # print('\nInner step', inner_step)

            def latent3DVar_cost_incremental():
                # ------------------------------------------------------
                # Background term
                # ------------------------------------------------------

                # If chi was a vector with shape (nnode*latent_dim, 1): J_b = 1/2 * (chi + chi_g)**T @ (chi + chi_g)
                # The formulation below is equivalent, faster, and works also for chi with shape (nnode, latent_dim)
                J_b = 1 /2 * torch.sum(torch.sum((corrected_chi_g + corrected_chi)*(corrected_chi_g + corrected_chi), axis=1)) # type: ignore
                # print('Jb', J_b.size())
                # print('Jb', J_b)

                # ------------------------------------------------------
                # Observation term
                # ------------------------------------------------------
                
                # J_o = 1/2 * (d - jH(jS**-1(jD(deltaz))))**T R**-1 (d - jH(jS**-1(jD(deltaz))))
                # 0) d = y - H(S**-1(D(zg)))
                # 1) S**-1(D(zg)), zg = zb + B_sqrt * chig
                # decoded_lat_vecg_dest = standardise_destandardise(
                #     input_fields = AE_model.decode(latent_bg + B_matrix_sqrt * corrected_chi_g),
                #     AE_props = AE_props,
                #     scalers_mean = AE_scalers_mean,
                #     scalers_std = AE_scalers_std,
                #     action = 'destandardise',
                #     device = device
                # )


                # 2) Apply observation operator H
                # HSDzg = interpolate_irregular_grid(grid_lats, grid_lons, decoded_lat_vecg_dest[:, obs_qty_idx], obs_lats, obs_lons)
                # innovation_vec = (obs_vec - HSDzg)

                # # Innovation vector at best current latent guess
                # H_xb = HS_decoder(latent_bg + B_matrix_sqrt * corrected_chi_g)
                # innovation_vec = obs_vec - H_xb

                # print(innovation_vec.size())
                # input('fd')

                # Perturbation in latent space
                # deltaz = B_matrix_sqrt*corrected_chi   # ensure this is mat-vec

                # print('tl tbd')
                # sleep(2)
                # Tangent linear operator applied to deltaz, expanded around the best guess latent state
                # _, J_HSD_deltaz = torch.autograd.functional.jvp(
                #     HS_decoder,
                #     (latent_bg + B_matrix_sqrt * corrected_chi_g,),
                #     (deltaz,), create_graph=True
                # )

                J_HSD_deltaz = torch.einsum('klmn,mn->kl', J_HSD, B_matrix_sqrt * corrected_chi)

                # print(J_HSD_deltaz.size())

                # print('tl done')
                # sleep(2)

                # _, J_HSD_deltaz = torch.func.jvp(
                #     HS_decoder,   # same callable as before
                #     (latent_bg + B_matrix_sqrt * corrected_chi_g,),
                #     (deltaz,),
                # )


                # # compute jH(jS**-1(jD(deltaz))))
                # # first (jS**-1(jD(deltaz)))) (trying simultaneously dealing with S and D)

                # deltaz = B_matrix_sqrt*corrected_chi
                # D_zg, jDdeltaz = torch.autograd.functional.jvp(destandardised_decoder_output, (current_best_guess_latent,), (deltaz,))

                # # now jacobian of observation operator

                # jHjSjDdeltaz = torch.autograd.functional.jvp(interpolate, (destandardised_decoder_output(current_best_guess_latent),), (destandardised_decoder_output(current_best_guess_latent + deltaz),))
                # # 3) Compute J_o with a similar speedup as J_b
                # J_o = 1 / 2 * torch.sum(torch.sum((innovation_vec - jHjSjDdeltaz) * R_matrix_inv * (innovation_vec - jHjSjDdeltaz), axis=1))

                # Observation term
                # J_o = 0.5 * torch.sum((innovation_vec - J_HSD_deltaz) * R_matrix_inv * (innovation_vec - J_HSD_deltaz), axis = 1)
                J_o = 1/2 * torch.sum(torch.sum((innovation_vec - J_HSD_deltaz) * R_matrix_inv * (innovation_vec - J_HSD_deltaz), axis = 1)) # type: ignore
                # print('J_o', J_o.size())

                # ------------------------------------------------------
                # Compute sum of terms and return them
                # ------------------------------------------------------

                J = torch.add(J_b, J_o)

                return J, J_b, J_o

            J, J_b, J_o = latent3DVar_cost_incremental()
            if sleepseconds > 0:
                print(f'Computed cost, sleeping {sleepseconds} seconds')
                sleep(sleepseconds)

            previous_chi = corrected_chi.clone().detach()  # Store current latent vector
            if sleepseconds > 0:
                print(f'Stored previous_chi, sleeping {sleepseconds} seconds')
                sleep(sleepseconds)
            
            # J.backward(retain_graph=True)#(retain_graph=True)  # Compute current gradient
            J.backward()
            if sleepseconds > 0:
                print(f'J.backward, sleeping {sleepseconds} seconds')
                sleep(sleepseconds)

            grad_J = corrected_chi.grad  # Get the gradient of the latent vector
            if sleepseconds > 0:
                print(f'grad_J, sleeping {sleepseconds} seconds')
                sleep(sleepseconds)

            grad_norm = torch.norm(grad_J, p=2).item()
            if sleepseconds > 0:
                print(f'grad_norm, sleeping {sleepseconds} seconds')
                sleep(sleepseconds)

            optimizer.step()  # Change the latent vector according to its current gradient
            if sleepseconds > 0:
                print(f'optimizer.step(), sleeping {sleepseconds} seconds')
                sleep(sleepseconds)

            optimizer.zero_grad()  # Clear the gradients for the next iteration
            if sleepseconds > 0:
                print(f'optimizer.zero_grad(), sleeping {sleepseconds} seconds')
                sleep(sleepseconds)


            all_J_inner.append(J.item())
            all_Jb_inner.append(J_b.item())
            all_Jo_inner.append(J_o.item())
            all_grad_inner.append(grad_norm)

            if sleepseconds > 0:
                print(f'append, sleeping {sleepseconds} seconds')
                sleep(sleepseconds)

            # Update best latent
            if J.item() < best_J_inner:
                best_J_inner = J.item()
                best_J_inner_step = inner_step
                best_grad_J = grad_norm
                best_latent_inner = outer_latent_detached + B_matrix_sqrt * previous_chi.detach()
                latents.append(best_latent_inner)

            else:
                 # Reduce LR if no improvement
                current_lr = optimizer.param_groups[0]['lr']
                new_lr = max(current_lr * factor_lr, minimum_lr)
                if new_lr == minimum_lr:
                    min_lr_counter += 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            # Print progress
            if inner_step == 1:
                 print(f"Initial J {J.item():.4f}, grad {grad_norm:.4f}")
            else:
                 print(f"Inner step {inner_step}, J {J.item():.4f}, ratio {all_J_inner[-1]/all_J_inner[-2]:.4f}, grad {grad_norm:.4f}")
            
            # Stopping criterion based on gradient reduction
            if inner_step >= 2 and all_grad_inner[-1]/all_grad_inner[0] < rtol_stop:
                print(f"Converged at inner step {inner_step}")
                break
            if inner_step >= 2:
                rel_change_J = abs(all_J_inner[-1] - all_J_inner[-2]) / all_J_inner[-2]
                if rel_change_J < 1e-5:
                    print(f"Converged by relative J change at inner step {inner_step}")
                    break

            if grad_norm < 1e-5:  # tiny gradients, essentially converged
                print(f"Gradient below threshold at inner step {inner_step}, stopping")
                break

            if min_lr_counter >= 3:
                print("Minimum LR used 3 times, stopping algorithm")
                break


            del J, J_b, J_o

        if sleepseconds > 0:
            print(f'Exiting inner loop, sleeping {sleepseconds} seconds')
            sleep(sleepseconds)

        # corrected_chi_g += corrected_chi
        corrected_chi_g += previous_chi
        outer_latent_detached = latent_bg + B_matrix_sqrt * corrected_chi_g.detach()


        # # --- Physical increment at this step using Jacobian of decoder
        # _, phys_increment = torch.autograd.functional.jvp(
        #         full_decoder,
        #         (outer_latent_detached,),
        #         ((B_matrix_sqrt*previous_chi).detach(),), create_graph=False
        #     )
        
        # # _, phys_increment = torch.func.jvp(
        # #         full_decoder,
        # #         (outer_latent_detached,),
        # #         ((B_matrix_sqrt*previous_chi).detach(),)
        # #     )
        # # store it
        # if 'phys_increments_outer' not in locals():
        #     phys_increments_outer = []
        # phys_increments_outer.append(phys_increment.detach())

        # Update outer latent with best inner result
        #best_latent_vec_overall = best_latent_inner.detach()
        best_latent_vec_overall = outer_latent_detached.detach()

        # Store outer loop stats
        all_J_outer.append(best_J_inner)
        all_Jb_outer.append(all_Jb_inner)
        all_Jo_outer.append(all_Jo_inner)
        all_grad_outer.append(best_grad_J)

        # Store full inner histories
        all_J.append(all_J_inner)
        all_Jb.append(all_Jb_inner)
        all_Jo.append(all_Jo_inner)
        all_grad.append(all_grad_inner)

        elapsed = time.time() - start_time 
        print(f"End of outer loop {outer_step}, best J {best_J_inner:.4f}, converged at step {best_J_inner_step}, best grad J ratio {(best_grad_J / all_grad_inner[0]):4f}, time elapsed {elapsed:.2f} seconds")
        
        if sleepseconds > 0:
            print(f'Sleeping {sleepseconds} seconds')
            sleep(sleepseconds)
        # sum physical increments for this outer loop
        # total_phys_increment = torch.stack(phys_increments_outer).sum(dim=0)
        # print(f'increment = {total_phys_increment}')
        # print('Ana. inc. at obs. loc. (interp. implemented by Janne)', interpolate_irregular_grid(grid_lats, grid_lons, total_phys_increment[:, obs_qty_idx], obs_lats, obs_lons))

    return {
        'out_latent': best_latent_vec_overall,
        'all_latents':latents,
        'all_J_outer': all_J_outer,
        'all_Jb_outer': all_Jb_outer,
        'all_Jo_outer': all_Jo_outer,
        'best_J': best_J_inner,
        'all_grad_J':all_grad,  
        'all_J':all_J,
        'all_Jb':all_Jb,
        'all_Jo':all_Jo,
        # 'increment': total_phys_increment
    }
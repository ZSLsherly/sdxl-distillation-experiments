import torch
def get_x0_from_noise(sample, model_output, timestep, alphas_cumprod):
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    # 0.0047 corresponds to the alphas_cumprod of the last timestep (999)
    # alpha_prod_t = (torch.ones_like(timestep) * 0.0047).reshape(-1, 1, 1, 1).double() 
    beta_prod_t = 1 - alpha_prod_t

    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    return pred_original_sample

def sample(
        noise, unet_added_conditions, model, vae, noise_scheduler, prompt_embed,
        device="cuda", num_step=1, conditioning_timestep=999
    ):
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    if num_step == 1:
        all_timesteps = [conditioning_timestep]
        step_interval = 0  
    elif num_step == 2:
        all_timesteps = [999, 499]
        step_interval = 500 
    elif num_step == 3:
        raise NotImplementedError() 
        all_timesteps = [999, 749, 499]
        step_interval = 250 
    elif num_step == 4:
        all_timesteps = [999, 749, 499, 249]
        step_interval = 250 
    else:
        raise NotImplementedError()
    
    DTYPE = prompt_embed.dtype
    
    for constant in all_timesteps:
        current_timesteps = torch.ones(len(prompt_embed), device=device, dtype=torch.long)  *constant
        eval_images = model(
            noise, current_timesteps, prompt_embed, added_cond_kwargs=unet_added_conditions
        ).sample

        eval_images = get_x0_from_noise(
            noise, eval_images, current_timesteps, alphas_cumprod
        ).float()

        next_timestep = current_timesteps - step_interval 
        noise = noise_scheduler.add_noise(
            eval_images, torch.randn_like(eval_images), next_timestep
        ).to(DTYPE)  

    eval_images = vae.decode(eval_images / vae.config.scaling_factor, return_dict=False)[0]
    eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    return eval_images
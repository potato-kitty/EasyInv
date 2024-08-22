

# here is an example of time-steps iteration
for i, t in enumerate(tqdm(reversed(model.scheduler.timesteps), desc="DDIM Inversion")):

    noise_pred = step_backward(model,latents,context,guidance_scale,t)
    noise_pred = noise_pred.requires_grad_(True)
    last_noise = noise_pred

    # above is an example of DDIM inversion, inject following codes right after it

    if (inject_steps + inject_len)*num_inference_steps > i > inject_steps*num_inference_steps:
        if i > 0:
            latents = 0.5*latents + 0.5*last_latent
    last_latent = latents
    
    # for SDXL, we recommand to set inject_steps = 0.85 and inject_len = 0.1 while inject_steps = 0.05 and inject_len = 0.2 for SDV1-4, 
    # num_inference_steps is the inference steps, which is usually set to 50 
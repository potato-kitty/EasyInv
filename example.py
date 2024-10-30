# here is an example of time-steps iteration
for i, t in enumerate(tqdm(reversed(model.scheduler.timesteps), desc="DDIM Inversion")):

    latents = DDIM_inversion(model,latents,context,guidance_scale,t)

    # above is an example of inversion, it could be basically any inversion algorithm, latents is the predicted latent for next step
    # inject following codes right after it to apply our method

    if (inject_steps + inject_len)*num_inference_steps > i > inject_steps*num_inference_steps:
        if i > 0:
            latents = eta*latents + (1 - eta)*last_latent
    last_latent = latents
    
    # for SDXL, we recommand to set eta = 0.8, inject_steps = 0.85 and inject_len = 0.1 while eta = 0.5, inject_steps = 0.05 and inject_len = 0.2 for SDV1-4,
    # num_inference_steps is the inference steps, which is usually set to 50 

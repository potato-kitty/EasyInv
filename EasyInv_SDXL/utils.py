import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import math
from torchviz import make_dot
from sklearn.cluster import KMeans
import os
from skimage import morphology
import copy
import torch.nn.functional as F
import random
import inspect


seed = 8888

os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

height = 512 
width = 512


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def SDXL(
    model,
    prompt: Union[str, List[str]] = None,
    real_img = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    inject_lens = 0.1, # 0.1
    fix_lens = 0, # 0
    inject_steps = 0.85, # 0.85
    fix_steps = 0, # 0
    inject_times = 1, # 1
    or_latent_idx = 0.8, # 0.8
    num_fix_itr = 0, # 0
    end_at = 1000000000,
    reconstract = False,
    timesteps: List[int] = None,
    denoising_end: Optional[float] = None,
    guidance_scale: float = 5.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image = None,
    ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    cross_attention_kwargs = None,
    guidance_rescale: float = 0.0,
    original_size: Optional[Tuple[int, int]] = None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    target_size: Optional[Tuple[int, int]] = None,
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_target_size: Optional[Tuple[int, int]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    **kwargs,
):

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None)

    # 0. Default height and width to unet
    #height = height or model.default_sample_size * model.vae_scale_factor
    #width = width or model.default_sample_size * model.vae_scale_factor

    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # 1. Check inputs. Raise error if not correct
    model.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt,
        negative_prompt_2,
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    model._guidance_scale = guidance_scale
    model._guidance_rescale = guidance_rescale
    model._clip_skip = clip_skip
    model._cross_attention_kwargs = cross_attention_kwargs
    model._denoising_end = denoising_end
    model._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = model._execution_device

    # 3. Encode input prompt
    lora_scale = (
        model.cross_attention_kwargs.get("scale", None) if model.cross_attention_kwargs is not None else None
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = model.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=model.do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=model.clip_skip
    )

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(model.scheduler, num_inference_steps, device, timesteps)

    # 5. Prepare latent variables
    if real_img is not None:
        latents = image2latent(model,real_img)
        start_latents = latents
    else:
        num_channels_latents = model.unet.config.in_channels
        latents = model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

    #print("latents: ",latents)

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    if model.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = model.text_encoder_2.config.projection_dim

    add_time_ids = model._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = model._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if model.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = model.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            model.do_classifier_free_guidance,
        )

    # 8. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)

    # 8.1 Apply denoising_end
    if (
        model.denoising_end is not None
        and isinstance(model.denoising_end, float)
        and model.denoising_end > 0
        and model.denoising_end < 1
    ):
        discrete_timestep_cutoff = int(
            round(
                model.scheduler.config.num_train_timesteps
                - (model.denoising_end * model.scheduler.config.num_train_timesteps)
            )
        )
        num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
        timesteps = timesteps[:num_inference_steps]

    # 9. Optionally get Guidance Scale Embedding
    timestep_cond = None
    if model.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(model.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = model.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=model.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    model._num_timesteps = len(timesteps)
    add_in = False
    with model.progress_bar(total=num_inference_steps) as progress_bar:
        if real_img is None:
            for i, t in enumerate(timesteps):

                if model.interrupt:
                    continue
                if not end_at > i:
                    break

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if model.do_classifier_free_guidance else latents

                latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=model.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if model.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + model.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if model.do_classifier_free_guidance and model.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=model.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        else:
            output_type = "latent"
            no_inject = 0
            no_add = True
            for i, t in enumerate(reversed(timesteps)):
                if model.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if model.do_classifier_free_guidance else latents

                latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = model.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=model.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if model.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + model.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if model.do_classifier_free_guidance and model.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=model.guidance_rescale)

                if (inject_steps + inject_lens)*num_inference_steps> i > inject_steps*num_inference_steps:
                    no_inject += 1
                    if i > 0:
                        latents = or_latent_idx*latents + (1 - or_latent_idx)*last_latent
                    print("add!")

                last_latent = latents
                latents,_ = next_step(model, noise_pred, int(t.item()), latents)

    if not output_type == "latent":
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = model.vae.dtype == torch.float16 

        if needs_upcasting:
            model.upcast_vae()
            latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(model.vae.config, "latents_mean") and model.vae.config.latents_mean is not None
        has_latents_std = hasattr(model.vae.config, "latents_std") and model.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / model.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / model.vae.config.scaling_factor

        image = model.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            model.vae.to(dtype=torch.float16)
    else:
        image = latents

    if not output_type == "latent":
        # apply watermark if available
        if model.watermark is not None:
            image = model.watermark.apply_watermark(image)

        image = model.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    model.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return image





def next_step(
    model,
    model_output: torch.FloatTensor,
    timestep: int,
    x: torch.FloatTensor,
    eta=0.,
    verbose=False
):
    """
    Inverse sampling for DDIM Inversion
    """
    if verbose:
        print("timestep: ", timestep)
    next_step = timestep
    timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999)
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[next_step]
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
    x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
    return x_next, pred_x0

@torch.no_grad()
def image2latent(model, image):
    # make sure the VAE is in float32 mode, as it overflows in float16
    needs_upcasting = model.vae.dtype == torch.float16 

    if needs_upcasting:
        model.upcast_vae()

    if type(image) is Image:
        image = np.array(image)
    image = torch.from_numpy(image).float() / 127.5 - 1 # transfer to pytorch tensor and norm
    image = image.permute(2, 0, 1).unsqueeze(0).to(device) # b,c,h,w    
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" 
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()


    latents = model.vae.encode(image)['latent_dist'].mean
    # cast back to fp16 if needed
    if needs_upcasting:
        model.vae.to(dtype=torch.float16)
    latents = latents * 0.18215
    if needs_upcasting:
        latents = latents.to(dtype=torch.float16)
    return latents

@torch.no_grad()
def text2image_ldm_stable(
    DDP_model,
    prompt: List[str],
    num_inference_steps: int = 50,
    real_img = None,
    full_img = None,
    generator: Optional[torch.Generator] = None,
):
    global device 
    device = DDP_model.device
    model = DDP_model

    if real_img is None and full_img is None: 
        img_out = SDXL(model,prompt,height=height,width=width,num_inference_steps=num_inference_steps,generator = generator)
    else:
        if full_img is not None:
            diff_step = 20
            print("num steps:",diff_step)
            inverse_img_full = SDXL(model,prompt[-1],height=height,width=width,num_inference_steps=diff_step,real_img=full_img,generator = generator)
            img_out = SDXL(model,prompt[-1],height=height,width=width,num_inference_steps=diff_step,latents=inverse_img_full,generator = generator)

    return img_out


import numpy as np
import torch
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import torch.nn.functional as F
from random import choice

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
    nxt_step = timestep
    timestep = min(timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps, 999)
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep] if timestep >= 0 else model.scheduler.final_alpha_cumprod
    alpha_prod_t_next = model.scheduler.alphas_cumprod[nxt_step]
    beta_prod_t = 1 - alpha_prod_t
    pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
    pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
    x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
    return x_next, pred_x0

@torch.no_grad()
def image2latent(model, image):
    DEVICE = device if torch.cuda.is_available() else torch.device("cpu")
    #if type(image) is Image:

    half = model.vae.dtype == torch.float16

    image = np.array(image)
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(0, 3, 1, 2).to(device)

    # input image density range [-1, 1]
    if half:
        model.vae.to(dtype=torch.float32)
    latents = model.vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    if half:
        model.vae.to(dtype=torch.float16)
        latents = latents.to(dtype=torch.float16)
    return latents


def diffusion_step(DDP_model, controller, latents, context, t, guidance_scale, low_resource=False, if_tst_L = False):
    model = DDP_model
    if low_resource:
        noise_pred_uncond,_,_,_ = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text,_,_,_ = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred,_,_,_ = model.unet(latents_input, t, encoder_hidden_states=context)
        noise_pred = noise_pred["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    if if_tst_L and noise_pred.shape[0] > 2:
        noise_pred[1] = noise_pred[0]
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents 


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image

def latent2image_tensor(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
            dtype=model.vae.dtype
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(device)
    return latent, latents

def step_forward(latents,model,context,guidance_scale,t):
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
    noise_pred,_,_,_ = model.unet(latent_model_input, t, encoder_hidden_states=context)
    noise_pred = noise_pred.sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    latents = model.scheduler.step(noise_pred, t, latents).prev_sample

    return latents, noise_pred

def step_forward_noise(latents,model,context,guidance_scale,t):
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
    noise_pred,_,_,_ = model.unet(latent_model_input, t, encoder_hidden_states=context)
    noise_pred = noise_pred.sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    #latents = model.scheduler.step(noise_pred, t, latents).prev_sample

    return noise_pred

def step_backward(model,latents,context,guidance_scale,t):

    model_inputs = torch.cat([latents] * 2)

    noise_pred,_,_,_ = model.unet(model_inputs, t, encoder_hidden_states=context)
    noise_pred = noise_pred.sample

    noise_pred_uncon, noise_pred_con = noise_pred.chunk(2, dim=0)
    noise_pred = noise_pred_uncon + guidance_scale * (noise_pred_con - noise_pred_uncon)

    return noise_pred

@torch.no_grad()
def invert(
    model,
    image,
    prompt,
    batch_size,
    num_inference_steps=50,
    guidance_scale=7.5,
    eta=0.0,
    loss_threshold = 0.0001,
    max_iter = 100,
    num_fix_itr = 0, # 6 for fix
    return_intermediates=False,
    opt_invert=False,
    **kwds):
    """
    invert a real image into noise map with determinisc DDIM inversion
    """

    prompt = prompt[-1]
    batch_size = 1
    # text embeddings
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    # define initial latents
    latents = image2latent(model,image)
    start_latents = latents


    # unconditional embedding for classifier free guidance
    if guidance_scale > 1.:

        max_length = text_input.input_ids.shape[-1]
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

        context = [uncond_embeddings, text_embeddings]
        context = torch.cat(context)

    # interative sampling
    model.scheduler.set_timesteps(num_inference_steps)

    latents_list = [latents]
    or_latent_idx = 0.5
    inject_steps = 0.05 #0.05
    inject_len = 0.2 #0.2
    no_inject = 0
    inject_times = 0
    for repeat in range(1):
        for i, t in enumerate(tqdm(reversed(model.scheduler.timesteps), desc="DDIM Inversion")):
            noise_pred = step_backward(model,latents,context,guidance_scale,t)
            noise_pred = noise_pred.requires_grad_(True)
            if not opt_invert:
                latents,_ = next_step(model, noise_pred, t, latents)
                if i > inject_steps*num_inference_steps and no_inject < inject_times:
                    no_inject += 1
                    latents = or_latent_idx*latents + (1 - or_latent_idx)*start_latents
                latents_list.append(latents)
                continue
            else:
                last_noise = noise_pred
                if (inject_steps + inject_len)*num_inference_steps > i > inject_steps*num_inference_steps:
                    print("add!")
                    if i > 0:
                        latents = or_latent_idx*latents + (1 - or_latent_idx)*last_latent
                for fix_itr in range(num_fix_itr):
                    if fix_itr == 0:
                        print("fix!")
                    if fix_itr > 0:
                        latents_tmp,_ = next_step(model, (noise_pred + last_noise)/2, t, latents)
                    else:
                        latents_tmp,_ = next_step(model, noise_pred, t, latents)
                    last_noise = noise_pred
                    noise_pred = step_forward_noise(latents_tmp,model,context,guidance_scale,t)

                last_latent = latents
                latents,_ = next_step(model, noise_pred, t, latents)
                latents_list.append(latents)
                continue

    model.vae.to(device)
    latents_list.reverse()
    return latents, latents_list, start_latents


@torch.no_grad()
def text2image_ldm_stable(
    DDP_model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    real_img = None,
    full_img = None,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    injection  = 0.1,
    is_switch = False,
    bboxes = None
):
    #num_inference_steps = 20

    model = DDP_model
    #model_for_inversion = copy.deepcopy(DDP_model)
    batch_size = len(prompt)
    if_tst_L = False
    global device
    device = model.device


    register_attention_control(model, controller,one_input=True)

    #real_img = None

    if real_img is not None:
        inverse_img,resize_img_latents_list,resize_start_latents = invert(model,real_img,prompt,batch_size,opt_invert=True)
    if full_img is not None:
        inverse_img_full,_,_ = invert(model,full_img,prompt,batch_size,opt_invert=True,num_inference_steps=num_inference_steps)

    #register_attention_control(model, controller,one_input=False)

    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    concate = True

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)


    latents[-1] = inverse_img_full[-1]

    model.scheduler.set_timesteps(num_inference_steps)


    for index, t in enumerate(tqdm(model.scheduler.timesteps)):
        latents = diffusion_step(DDP_model, controller, latents, context, t, guidance_scale, low_resource, if_tst_L=if_tst_L)
 
    image = latent2image(model.vae, latents)

    return image, latent


def register_attention_control(model, controller, no_switch = False, one_input = False):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None, one = one_input):
            if context is not None and one:
                context = context[-2:]
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            scaler = 0.5
            scaler_min = 0.55

            or_device = q.device
            q = q
            k = k

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            q = q.to(or_device)
            k = k.to(or_device)
            sim = sim.to(or_device)

            attn = sim

            #
            if len(context.shape) < 3:
                context = context.unsqueeze(0)

            or_device = attn.device
            attn = attn
            attn = attn.softmax(dim=-1)
            attn = attn.to(or_device)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)

            out = self.reshape_batch_dim_to_heads(out)
            out = to_out(out)
            torch.cuda.empty_cache()
            return out, attn

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

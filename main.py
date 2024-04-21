import pyrallis
import torch
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from src.config import RunConfig
from src.utils.enums_utils import model_type_to_size, is_stochastic

def create_noise_list(model_type, length, generator=None):
    img_size = model_type_to_size(model_type)
    VQAE_SCALE = 8
    latents_size = (1, 4, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
    return [randn_tensor(latents_size, dtype=torch.float16, device=torch.device("cuda:0"), generator=generator) for i in range(length)]

@pyrallis.wrap()
def main(cfg: RunConfig):
    run(cfg)

def run(init_image: Image,
        prompt: str,
        cfg: RunConfig,
        pipe_inversion,
        pipe_inference,
        latents = None,
        edit_prompt = None,
        edit_cfg = 1.0,
        noise = None,
        do_reconstruction = True):
    
    generator = torch.Generator().manual_seed(cfg.seed)

    if is_stochastic(cfg.scheduler_type):
        if latents is None:
            noise = create_noise_list(cfg.model_type, cfg.num_inversion_steps, generator=generator)
        pipe_inversion.scheduler.set_noise_list(noise)
        pipe_inference.scheduler.set_noise_list(noise)

    pipe_inversion.cfg = cfg
    pipe_inference.cfg = cfg
    all_latents = None

    if latents is None:
        print("Inverting...")
        res = pipe_inversion(prompt = prompt,
                        num_inversion_steps = cfg.num_inversion_steps,
                        num_inference_steps = cfg.num_inference_steps,
                        generator = generator,
                        image = init_image,
                        guidance_scale = cfg.guidance_scale,
                        strength = cfg.inversion_max_step,
                        denoising_start = 1.0-cfg.inversion_max_step,
                        num_renoise_steps = cfg.num_renoise_steps)
        latents = res[0][0]
        all_latents = res[1]
    
    inv_latent = latents.clone()

    if do_reconstruction:
        print("Generating...")
        edit_prompt = prompt if edit_prompt is None else edit_prompt
        guidance_scale = edit_cfg
        img = pipe_inference(prompt = edit_prompt,
                            num_inference_steps = cfg.num_inference_steps,
                            negative_prompt = prompt,
                            image = latents,
                            strength = cfg.inversion_max_step,
                            denoising_start = 1.0-cfg.inversion_max_step,
                            guidance_scale = guidance_scale).images[0]
    else:
        img = None
                    
    return img, inv_latent, noise, all_latents

if __name__ == "__main__":
    main()

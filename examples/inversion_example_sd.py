import torch
from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_type = Model_Type.SD15
scheduler_type = Scheduler_Type.DDIM
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

input_image = Image.open("example_images/lion.jpeg").convert("RGB").resize((512, 512))
prompt = "a lion in the field"

config = RunConfig(model_type = model_type,
                    num_inference_steps = 50,
                    num_inversion_steps = 50,
                    num_renoise_steps = 1,
                    scheduler_type = scheduler_type,
                    perform_noise_correction = False,
                    seed = 7865)

_, inv_latent, _, all_latents = invert(input_image,
                                       prompt,
                                       config,
                                       pipe_inversion=pipe_inversion,
                                       pipe_inference=pipe_inference,
                                       do_reconstruction=False)

rec_image = pipe_inference(image = inv_latent,
                           prompt = prompt,
                           strength=1.0,
                           num_inference_steps = config.num_inference_steps,
                           guidance_scale = 1.0).images[0]

rec_image.save("lion_reconstructed.jpg")

# ReNoise: Real Image Inversion Through Iterative Noising
> **Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, Daniel Cohen-Or**
>
> Recent advancements in text-guided diffusion models have unlocked powerful image manipulation capabilities. However, applying these methods to real images necessitates the inversion of the images into the domain of the pretrained diffusion model. Achieving faithful inversion remains a challenge, particularly for more recent models trained to generate images with a small number of denoising steps. In this work, we introduce an inversion method with a high quality-to-operation ratio, enhancing reconstruction accuracy without increasing the number of operations. Building on reversing the diffusion sampling process, our method employs an iterative renoising mechanism at each inversion sampling step. This mechanism refines the approximation of a predicted point along the forward diffusion trajectory, by iteratively applying the pretrained diffusion model, and averaging these predictions. We evaluate the performance of our ReNoise technique using various sampling algorithms and models, including recent accelerated diffusion models. Through comprehensive evaluations and comparisons, we show its effectiveness in terms of both accuracy and speed. Furthermore, we confirm that our method preserves editability by demonstrating text-driven image editing on real images.

<a href="https://garibida.github.io/ReNoise-Inversion/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2403.14602"><img src="https://img.shields.io/badge/arXiv-ReNoise-b31b1b.svg" height=20.5></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/garibida/ReNoise-Inversion)

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>
</p>

## Description
Official implementation of our ReNoise paper.

## Environment Setup
Our code builds on the requirement of the `diffusers` library. To set up the environment, please run:
```
conda env create -f environment.yaml
conda activate renoise_inversion
```
or install requirements:
```
pip install -r requirements.txt
```
## Demo
To run a local demo of the project, run the following:
```
gradio gradio_app.py
```

## Usage
There are three examples of how to use the inversion in Stable Diffusion, SDXL, and SDXL Turbo. You can find these examples in the `examples/` directory.

### Inversion

We have created a diffusers pipe to perform the inversion. You can use the following code to use ReNoise in your project:

```
from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert

model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

input_image = Image.open("example_images/lion.jpeg").convert("RGB").resize((512, 512))
prompt = "a lion in the field"

config = RunConfig(model_type = model_type,
                    scheduler_type = scheduler_type)

rec_img, inv_latent, noise, all_latents = invert(input_image,
                                                 prompt,
                                                 config,
                                                 pipe_inversion=pipe_inversion,
                                                 pipe_inference=pipe_inference,
                                                 do_reconstruction=False)
```

You can controll the inversion paremeters using the following attributes in the `RunConfig`:
- `num_inference_steps` - Number of denoise steps.
- `num_inversion_steps` - Number of inversion steps.
- `guidance_scale` - Guidence scale during the inversion.
- `num_renoise_steps` - Number of ReNoise steps.
- `max_num_renoise_steps_first_step` - Max number of ReNoise steps when `T<250`
- `inversion_max_step` - Inversion strength. The number of denoising steps depends on the amount of noise initially added. When strength is `1.0`, the image will be inverted to complete noise and the denoising process will run for the full number of steps. When strength is `0.5`, the image will be inverted to half noise and the denoising process will run for half of the steps.
- `num_inference_steps` - Number of denoise steps
- `average_latent_estimations` - Perform estimations averaging.
- `average_first_step_range` - Averaging range when `T<250`. The value is tuple, for example `(0, 5)`.
- `average_first_step_range` - Averaging range when `T>250`. The value is tuple, for example `(8, 10)`.
- `noise_regularization_lambda_ac` - Noise regularization pairwise lambda.
- `noise_regularization_lambda_kl` - Noise regularization patch KL divergence lambda.
- `perform_noise_correction` - Perform noise correction.

In case of stochastic sampler add the following to use the same \epsilon_t as in the inversion process.
```
pipe_inference.scheduler.set_noise_list(noise)
```

### Edit
To edit images using ReNoise you can use the following code:
```
from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert

model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

input_image = Image.open("example_images/lion.jpeg").convert("RGB").resize((512, 512))
prompt = "a lion in the field"

config = RunConfig(model_type = model_type,
                    scheduler_type = scheduler_type)

edit_img, inv_latent, noise, all_latents = invert(input_image,
                                                 prompt,
                                                 config,
                                                 pipe_inversion=pipe_inversion,
                                                 pipe_inference=pipe_inference,
                                                 do_reconstruction=True,
                                                 edit_prompt="a tiger in the field"
                                                 )

edit_img.save("result.png")
```

## Acknowledgements 
This code builds on the code from the [diffusers](https://github.com/huggingface/diffusers) library. In addition, we 
borrow code from the following repositories: 
- [Pix2PixZero](https://github.com/pix2pixzero/pix2pix-zero) for noise regularization.
- [sdxl_inversions](https://github.com/cloneofsimo/sdxl_inversions) for initial implementation of DDIM inversion in SDXL.


## Citation
If you use this code for your research, please cite the following work: 
```
@misc{garibi2024renoise,
      title={ReNoise: Real Image Inversion Through Iterative Noising}, 
      author={Daniel Garibi and Or Patashnik and Andrey Voynov and Hadar Averbuch-Elor and Daniel Cohen-Or},
      year={2024},
      eprint={2403.14602},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
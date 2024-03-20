# ReNoise: Real Image Inversion Through Iterative Noising

> Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, Daniel Cohen-Or  
>
> Recent advancements in text-guided diffusion models have unlocked powerful image manipulation capabilities. However, applying these methods to real images necessitates the inversion of the images into the domain of the pretrained diffusion model. Achieving faithful inversion remains a challenge, particularly for more recent models trained to generate images with a small number of denoising steps. In this work, we introduce an inversion method with a high quality-to-operation ratio, enhancing reconstruction accuracy without increasing the number of operations. Building on reversing the diffusion sampling process, our method employs an iterative renoising mechanism at each inversion sampling step. This mechanism refines the approximation of a predicted point along the forward diffusion trajectory, by iteratively applying the pretrained diffusion model, and averaging these predictions. We evaluate the performance of our ReNoise technique using various sampling algorithms and models, including recent accelerated diffusion models. Through comprehensive evaluations and comparisons, we show its effectiveness in terms of both accuracy and speed. Furthermore, we confirm that our method preserves editability by demonstrating text-driven image editing on real images.

<!-- <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv.svg" height=22.5></a> -->
<a href="[https://garibida.github.io/cross-image-attention/](https://garibida.github.io/ReNoise-Inversion/)"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=20.5></a>

<p align="center">
<img src="docs/teaser.jpg" width="90%"/>  
<br>
Our ReNoise inversion technique can be applied to various diffusion models, including recent few-step ones. This figure illustrates the performance of our method with SDXL Turbo and LCM models, showing its effectiveness compared to DDIM inversion. Additionally, we demonstrate that the quality of our inversions allows prompt-driven editing. As illustrated on the right, our approach also allows for prompt-driven image edits.
</p>

# Code Coming Soon!

## Citation
If you use this code for your research, please cite the following work: 
```
@misc{
}

from diffusers import LCMScheduler
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
import torch
from typing import List, Optional, Tuple, Union
import numpy as np

class LCMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    denoised: Optional[torch.FloatTensor] = None

class MyLCMScheduler(LCMScheduler):

    def set_noise_list(self, noise_list):
        self.noise_list = noise_list
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LCMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        self._init_step_index(timestep)

        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        # 4. Compute the predicted original sample x_0 based on the model parameterization
        if self.config.prediction_type == "epsilon":  # noise-prediction
            predicted_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.config.prediction_type == "sample":  # x-prediction
            predicted_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":  # v-prediction
            predicted_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction` for `LCMScheduler`."
            )

        # 5. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            predicted_original_sample = self._threshold_sample(predicted_original_sample)
        elif self.config.clip_sample:
            predicted_original_sample = predicted_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 6. Denoise model output using boundary conditions
        denoised = c_out * predicted_original_sample + c_skip * sample

        # 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        # Noise is not used on the final timestep of the timestep schedule.
        # This also means that noise is not used for one-step sampling.
        if self.step_index != self.num_inference_steps - 1:
            noise = self.noise_list[self.step_index]
            prev_sample = alpha_prod_t_prev.sqrt() * denoised + beta_prod_t_prev.sqrt() * noise
        else:
            prev_sample = denoised

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, denoised)

        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=denoised)
    

    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[LCMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] or `tuple`.
        Returns:
            [`~schedulers.scheduling_utils.LCMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_lcm.LCMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        self._init_step_index(timestep)

        # 1. get previous step value
        prev_step_index = self.step_index + 1
        if prev_step_index < len(self.timesteps):
            prev_timestep = self.timesteps[prev_step_index]
        else:
            prev_timestep = timestep

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 3. Get scalings for boundary conditions
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)

        if self.step_index != self.num_inference_steps - 1:
            c_skip_actual = c_skip * alpha_prod_t_prev.sqrt()
            c_out_actual = c_out * alpha_prod_t_prev.sqrt()
            noise = self.noise_list[self.step_index] * beta_prod_t_prev.sqrt()
        else:
            c_skip_actual = c_skip
            c_out_actual = c_out
            noise = 0


        dem = c_out_actual / (alpha_prod_t.sqrt()) + c_skip
        eps_mul = beta_prod_t.sqrt() * c_out_actual / (alpha_prod_t.sqrt())

        prev_sample = (sample + eps_mul * model_output - noise) / dem

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample, prev_sample)

        return LCMSchedulerOutput(prev_sample=prev_sample, denoised=prev_sample)
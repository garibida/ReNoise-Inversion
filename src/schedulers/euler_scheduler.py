from diffusers import EulerAncestralDiscreteScheduler
from diffusers.utils import BaseOutput
import torch
from typing import List, Optional, Tuple, Union
import numpy as np

class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
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
    pred_original_sample: Optional[torch.FloatTensor] = None

class MyEulerAncestralDiscreteScheduler(EulerAncestralDiscreteScheduler):
    def set_noise_list(self, noise_list):
        self.noise_list = noise_list

    def get_noise_to_remove(self):
        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5

        return self.noise_list[self.step_index] * sigma_up\
        
    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Union[float, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """

        self._init_step_index(timestep.view((1)))
        return EulerAncestralDiscreteScheduler.scale_model_input(self, sample, timestep)

    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
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
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        self._init_step_index(timestep.view((1)))

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        # derivative = (sample - pred_original_sample) / sigma
        derivative = model_output

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up

        prev_sample = prev_sample + self.noise_list[self.step_index] * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
    
    def step_and_update_noise(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        expected_prev_sample: torch.FloatTensor,
        optimize_epsilon_type: bool = False,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
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
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        self._init_step_index(timestep.view((1)))

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        # derivative = (sample - pred_original_sample) / sigma
        derivative = model_output

        dt = sigma_down - sigma

        prev_sample = sample + derivative * dt

        device = model_output.device
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up

        if sigma_up > 0:
            req_noise = (expected_prev_sample - prev_sample) / sigma_up
            if not optimize_epsilon_type:
                self.noise_list[self.step_index] = req_noise
            else:
                for i in range(10):
                    n = torch.autograd.Variable(self.noise_list[self.step_index].detach().clone(), requires_grad=True)
                    loss = torch.norm(n - req_noise.detach())
                    loss.backward()
                    self.noise_list[self.step_index] -= n.grad.detach() * 1.8


        prev_sample = prev_sample + self.noise_list[self.step_index] * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
    
    def inv_step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
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
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.

        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        self._init_step_index(timestep.view((1)))

        sigma = self.sigmas[self.step_index]

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[self.step_index]
        sigma_to = self.sigmas[self.step_index+1]
        # sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2).abs() / sigma_from**2) ** 0.5
        # sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
        sigma_down = sigma_to**2 / sigma_from

        # 2. Convert to an ODE derivative
        # derivative = (sample - pred_original_sample) / sigma
        derivative = model_output

        dt = sigma_down - sigma
        # dt = sigma_down - sigma_from

        prev_sample = sample - derivative * dt

        device = model_output.device
        # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
        # prev_sample = prev_sample + noise * sigma_up

        prev_sample = prev_sample - self.noise_list[self.step_index] * sigma_up

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return EulerAncestralDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )
    
    def get_all_sigmas(self) -> torch.FloatTensor:
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        return torch.from_numpy(sigmas)
    
    def add_noise_off_schedule(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.get_all_sigmas()
        sigmas = sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            timesteps = timesteps.to(original_samples.device)

        step_indices = 1000 - int(timesteps.item())

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples
    
    # def update_noise_for_friendly_inversion(
    #     self,
    #     model_output: torch.FloatTensor,
    #     timestep: Union[float, torch.FloatTensor],
    #     z_t: torch.FloatTensor,
    #     z_tp1: torch.FloatTensor,
    #     return_dict: bool = True,
    # ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
    #     if (
    #         isinstance(timestep, int)
    #         or isinstance(timestep, torch.IntTensor)
    #         or isinstance(timestep, torch.LongTensor)
    #     ):
    #         raise ValueError(
    #             (
    #                 "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
    #                 " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
    #                 " one of the `scheduler.timesteps` as a timestep."
    #             ),
    #         )

    #     if not self.is_scale_input_called:
    #         logger.warning(
    #             "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
    #             "See `StableDiffusionPipeline` for a usage example."
    #         )

    #     self._init_step_index(timestep.view((1)))

    #     sigma = self.sigmas[self.step_index]

    #     sigma_from = self.sigmas[self.step_index]
    #     sigma_to = self.sigmas[self.step_index+1]
    #     # sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    #     sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2).abs() / sigma_from**2) ** 0.5
    #     # sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    #     sigma_down = sigma_to**2 / sigma_from

    #     # 2. Conv = (sample - pred_original_sample) / sigma
    #     derivative = model_output

    #     dt = sigma_down - sigma
    #     # dt = sigma_down - sigma_from

    #     prev_sample = z_t - derivative * dt

    #     if sigma_up > 0:
    #         self.noise_list[self.step_index] = (prev_sample - z_tp1) / sigma_up

    #     prev_sample = prev_sample - self.noise_list[self.step_index] * sigma_up
        

    #     if not return_dict:
    #         return (prev_sample,)

    #     return EulerAncestralDiscreteSchedulerOutput(
    #         prev_sample=prev_sample, pred_original_sample=None
    #     )

    
    # def step_friendly_inversion(
    #     self,
    #     model_output: torch.FloatTensor,
    #     timestep: Union[float, torch.FloatTensor],
    #     sample: torch.FloatTensor,
    #     generator: Optional[torch.Generator] = None,
    #     return_dict: bool = True,
    #     expected_next_sample: torch.FloatTensor = None,
    # ) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:
    #     """
    #     Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    #     process from the learned model outputs (most often the predicted noise).

    #     Args:
    #         model_output (`torch.FloatTensor`):
    #             The direct output from learned diffusion model.
    #         timestep (`float`):
    #             The current discrete timestep in the diffusion chain.
    #         sample (`torch.FloatTensor`):
    #             A current instance of a sample created by the diffusion process.
    #         generator (`torch.Generator`, *optional*):
    #             A random number generator.
    #         return_dict (`bool`):
    #             Whether or not to return a
    #             [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or tuple.

    #     Returns:
    #         [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
    #             If return_dict is `True`,
    #             [`~schedulers.scheduling_euler_ancestral_discrete.EulerAncestralDiscreteSchedulerOutput`] is returned,
    #             otherwise a tuple is returned where the first element is the sample tensor.

    #     """

    #     if (
    #         isinstance(timestep, int)
    #         or isinstance(timestep, torch.IntTensor)
    #         or isinstance(timestep, torch.LongTensor)
    #     ):
    #         raise ValueError(
    #             (
    #                 "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
    #                 " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
    #                 " one of the `scheduler.timesteps` as a timestep."
    #             ),
    #         )

    #     if not self.is_scale_input_called:
    #         logger.warning(
    #             "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
    #             "See `StableDiffusionPipeline` for a usage example."
    #         )

    #     self._init_step_index(timestep.view((1)))

    #     sigma = self.sigmas[self.step_index]

    #     # Upcast to avoid precision issues when computing prev_sample
    #     sample = sample.to(torch.float32)

    #     # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    #     if self.config.prediction_type == "epsilon":
    #         pred_original_sample = sample - sigma * model_output
    #     elif self.config.prediction_type == "v_prediction":
    #         # * c_out + input * c_skip
    #         pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    #     elif self.config.prediction_type == "sample":
    #         raise NotImplementedError("prediction_type not implemented yet: sample")
    #     else:
    #         raise ValueError(
    #             f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
    #         )

    #     sigma_from = self.sigmas[self.step_index]
    #     sigma_to = self.sigmas[self.step_index + 1]
    #     sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    #     sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    #     # 2. Convert to an ODE derivative
    #     # derivative = (sample - pred_original_sample) / sigma
    #     derivative = model_output

    #     dt = sigma_down - sigma

    #     prev_sample = sample + derivative * dt

    #     device = model_output.device
    #     # noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)
    #     # prev_sample = prev_sample + noise * sigma_up

    #     if sigma_up > 0:
    #         self.noise_list[self.step_index] = (expected_next_sample - prev_sample) / sigma_up

    #     prev_sample = prev_sample + self.noise_list[self.step_index] * sigma_up

    #     # Cast sample back to model compatible dtype
    #     prev_sample = prev_sample.to(model_output.dtype)

    #     # upon completion increase step index by one
    #     self._step_index += 1

    #     if not return_dict:
    #         return (prev_sample,)

    #     return EulerAncestralDiscreteSchedulerOutput(
    #         prev_sample=prev_sample, pred_original_sample=pred_original_sample
    #     )
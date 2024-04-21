from dataclasses import dataclass

from src.eunms import Model_Type, Scheduler_Type

@dataclass
class RunConfig:
    model_type : Model_Type = Model_Type.SDXL_Turbo

    scheduler_type : Scheduler_Type = Scheduler_Type.EULER

    seed: int = 7865

    num_inference_steps: int = 4

    num_inversion_steps: int = 4

    guidance_scale: float = 0.0

    num_renoise_steps: int = 9

    max_num_renoise_steps_first_step: int = 5

    inversion_max_step: float = 1.0

    # Average Parameters

    average_latent_estimations: bool = True

    average_first_step_range: tuple = (0, 5)

    average_step_range: tuple = (8, 10)

    # Noise Regularization

    noise_regularization_lambda_ac: float = 20.0

    noise_regularization_lambda_kl: float = 0.065
    
    noise_regularization_num_reg_steps: int = 4

    noise_regularization_num_ac_rolls: int = 5

    # Noise Correction

    perform_noise_correction: bool = True

    def __post_init__(self):
        pass
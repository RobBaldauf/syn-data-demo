"""Synthetic data generation demo script for ControlNet."""
from pathlib import Path
from cldm.hack import disable_verbosity, enable_sliced_attention
import hydra
from omegaconf import DictConfig
from pathlib import Path

from use_cases.mri_image_generation import MriImageGeneration


@hydra.main(config_path="../", config_name="config",version_base=None)
def main(cfg: DictConfig)-> None:
    """Main function to run for the controlnet synthetic data generation script.
    
    Args:
        cfg: Global hydra config object containing all parameters.
    """
    disable_verbosity()
    if cfg.enable_save_memory:  
        enable_sliced_attention()
    if cfg.use_case == "mri_image_gen":
        generator = MriImageGeneration(
            save_memory=cfg.enable_save_memory, 
            model_path= Path(cfg.canny.model_path),
            checkpoint_path=Path(cfg.canny.checkpoint_path),
            device="cuda" if cfg.enable_cuda else "cpu",
            low_threshold=cfg.canny.low_threshold,
            high_threshold=cfg.canny.high_threshold,
        )
    generator.gradio_ui().launch(server_name='0.0.0.0')

if __name__ == "__main__":
    main()

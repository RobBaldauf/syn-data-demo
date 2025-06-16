
"""Synthetic data generation demo script for ControlNet."""

from functools import partial
from pathlib import Path
from typing import List

import gradio as gr
import einops

import numpy as np
import torch
import random
from annotator.util import resize_image, HWC3
from pytorch_lightning import seed_everything

from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from image_utils import take_luminance_from_first_chroma_from_second

class MriImageGeneration:
    """Class for generating MRI images using ControlNet with Canny edge maps."""

    def __init__(self, save_memory:bool, model_path:Path, checkpoint_path:Path, device:str='cpu')->None:
        """Initialize the MRI image generation model with ControlNet and Canny edge maps.
        
        Args:
            save_memory: Whether to save memory by using low VRAM mode.
            model_path: Path to the ControlNet model.
            checkpoint_path: Path to the model checkpoint.
            device: Device to run the model on ('cpu' or 'cuda').
        
        """
        self._save_memory = save_memory
        self._device = device
        self._model = create_model(model_path).cpu()
        self._model.load_state_dict(load_state_dict(checkpoint_path, location=self._device))
        if self._device == 'cuda':
            model = model.cuda()
        self._ddim_sampler = DDIMSampler(self._model, device=self._device)
        self._apply_canny = CannyDetector()

    def process_sample(self, input_image:np.ndarray, prompt:str,  a_prompt:str, n_prompt:str, num_samples:int, image_resolution:int, ddim_steps:int, guess_mode:bool, strength:int, scale:float, seed:int, eta:float,low_threshold:int,high_threshold:int)->List[np.ndarray]:
        """Process a sample image and generate MRI images using ControlNet with Canny edge maps.
        
        Args:
            input_image: Input image to process.
            prompt: Text prompt for image generation.
            a_prompt: Additional prompt for image generation.
            n_prompt: Negative prompt for image generation.
            num_samples: Number of samples to generate.
            image_resolution: Resolution of the generated images.
            ddim_steps: Number of DDIM steps for sampling.
            guess_mode: Whether to use guess mode for ControlNet.
            strength: Strength of the control signal.
            scale: Guidance scale for the model.
            seed: Random seed for reproducibility.
            eta: Eta value for DDIM sampling.
            low_threshold: Low threshold for Canny edge detection.
            high_threshold: High threshold for Canny edge detection.
        """
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self._apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            if self._device == 'cuda':
                control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            else:
                control = torch.from_numpy(detected_map.copy()).float() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if self._save_memory:
                self._model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self._model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self._model.get_learned_conditioning([n_prompt] * num_samples)]}
            H, W, _ = input_image.shape
            shape = (4, H // 8, W // 8)

            if self._save_memory:
                self._model.low_vram_shift(is_diffusing=True)

            self._model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, _ = self._ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if self._save_memory:
                self._model.low_vram_shift(is_diffusing=False)

            x_samples = self._model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
            index = -1
        return [
            input_image, 
            255 - detected_map,
            take_luminance_from_first_chroma_from_second(resize_image(HWC3(input_image), image_resolution), results[index], mode="lab")
        ]
    
    def gradio_ui(self)->gr.Blocks:
        """Create a Gradio UI for the MRI image generation model.
        
        Returns:
            Gradio Blocks object containing the UI components.
        """
        block = gr.Blocks().queue()
        with block:
            with gr.Row():
                gr.Markdown("## MRI Image Generation with Control Net and Canny Edge Maps")
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(source='upload', type="numpy")
                    prompt = gr.Textbox(label="Prompt", value="mri brain scan", placeholder="Enter your prompt here")
                    run_button = gr.Button(label="Run")
                    with gr.Accordion("Advanced options", open=False):
                        num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                        image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                        strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                        guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                        low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=50, step=1)
                        high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=100, step=1)
                        ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=10, step=1)
                        scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=1)
                        eta = gr.Number(label="eta (DDIM)", value=0.0)
                        a_prompt = gr.Textbox(label="Added Prompt", value='good quality')
                        n_prompt = gr.Textbox(label="Negative Prompt",
                                            value='animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
                with gr.Column():
                    result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
            ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
            run_button.click(fn=partial(MriImageGeneration.process_sample,self), inputs=ips, outputs=[result_gallery])
        return block
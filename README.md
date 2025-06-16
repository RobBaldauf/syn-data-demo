# ControlNet Synthetic Data Generation Demos
Based on [ControlNet 1.0](https://github.com/lllyasviel/ControlNet)

# Usage

First create a new conda environment

    conda env create -f environment.yaml
    conda activate control

All models and detectors can be downloaded from [our Hugging Face page](https://huggingface.co/lllyasviel/ControlNet). Make sure that SD models are put in "models/" and detectors are put in "annotator/ckpts". Make sure that you download all necessary pretrained weights and detector models from that Hugging Face page, including HED edge detection model, Midas depth estimation model, Openpose, and so on. 

## ControlNet for Synthetic MRI data generation 

Stable Diffusion 1.5 + ControlNet (using simple Canny edge detection)

    python src/run.py use_case=mri_image_gen 

This will spawn a Gradio app also allows you to change the Canny edge thresholds.

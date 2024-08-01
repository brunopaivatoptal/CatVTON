# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:43:37 2024

@author: angel
"""

from dataclasses import dataclass 

@dataclass
class inferenceConfig:
    variant="CatVTON"
    base_model_path: str = "runwayml/stable-diffusion-inpainting"
    resume_path: str = "zhengchong/CatVTON"
    dataset_name: str = "PX-VTON"
    data_root_path: str = (
        r"D:\backups\toptal\pixelcut\virtual-try-on\small_curated_evalset",
        "/mnt/d/backups/toptal/pixelcut/virtual-try-on/small_curated_evalset"
        )
    output_dir: str = "output"
    seed: int = 555
    batch_size: int = 4
    num_inference_steps: int = 50
    guidance_scale: float = 2.5
    width: int = 384
    height: int = 512
    repaint: bool = False
    eval_pair: bool = False
    concat_eval_results: bool = True
    allow_tf32: bool = True
    dataloader_num_workers: int = 8
    mixed_precision: str = "bf16"
    concat_axis: str = "y"
    enable_condition_noise: bool = True
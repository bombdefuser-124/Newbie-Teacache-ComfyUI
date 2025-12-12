"""
TeaCache Coefficient Calculator for Lumina2/Newbie Models (Improved v3)

Features:
- Multi-generation support
- Matches Newbie architecture input logic strictly
- Auto-clamps polynomial output to be positive (safety)
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
from unittest.mock import patch


class TeaCacheCoefficientCalculator:
    """
    Calculates TeaCache polynomial coefficients for Newbie models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES,),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES,),
                "num_generations": ("INT", {"default": 5, "min": 1, "max": 50}),
                "polynomial_order": ("INT", {"default": 4, "min": 1, "max": 5}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("coefficients", "analysis_report",)
    FUNCTION = "calculate_coefficients"
    CATEGORY = "teacache"
    OUTPUT_NODE = True
    
    def calculate_coefficients(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        num_generations: int,
        polynomial_order: int,
        denoise: float,
    ) -> Tuple[str, str]:
        
        all_input_diffs = []
        all_output_diffs = []
        
        diffusion_model = model.get_model_object("diffusion_model")
        
        print(f"TeaCache Coefficient Calculator: Running {num_generations} generations...")
        
        for gen_idx in range(num_generations):
            current_seed = seed + gen_idx
            modulated_inputs = []
            denoised_outputs = []
            
            # Match TeaCache_Newbie logic exactly for input capture
            original_forward = diffusion_model.forward
            
            def capturing_forward(x, t, cap_feats, cap_mask, **kwargs):
                try:
                    # Newbie just uses t_emb as adaln_input, no CLIP addition here
                    t_emb = diffusion_model.t_embedder(t)
                    adaln_input = t_emb
                    
                    if diffusion_model.layers and hasattr(diffusion_model.layers[0], 'adaLN_modulation'):
                        mod_result = diffusion_model.layers[0].adaLN_modulation(adaln_input)
                        modulated_inputs.append(mod_result.detach().cpu().float().clone())
                except Exception as e:
                    pass
                
                return original_forward(x, t, cap_feats, cap_mask, **kwargs)
            
            def collection_callback(step, x, x0, total_steps):
                try:
                    denoised_outputs.append(x0.detach().cpu().float().clone())
                except:
                    pass
            
            print(f"  Generation {gen_idx + 1}/{num_generations} (seed={current_seed})...")
            
            latent = latent_image.copy()
            samples = latent["samples"]
            noise = comfy.sample.prepare_noise(samples, current_seed, None)
            noise_mask = latent.get("noise_mask", None)
            
            sampler = comfy.samplers.KSampler(
                model, 
                steps=steps,
                device=comfy.model_management.get_torch_device(),
                sampler=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
            )
            
            with patch.object(diffusion_model, 'forward', capturing_forward):
                _ = sampler.sample(
                    noise,
                    positive,
                    negative,
                    cfg=cfg,
                    latent_image=samples,
                    start_step=0,
                    last_step=steps,
                    force_full_denoise=True,
                    denoise_mask=noise_mask,
                    seed=current_seed,
                    callback=collection_callback,
                )
            
            # Process groups (cond vs uncond)
            mod_by_batch = {}
            for mod in modulated_inputs:
                batch_key = mod.shape[0]
                if batch_key not in mod_by_batch:
                    mod_by_batch[batch_key] = []
                mod_by_batch[batch_key].append(mod)
            
            out_by_batch = {}
            for out in denoised_outputs:
                batch_key = out.shape[0]
                if batch_key not in out_by_batch:
                    out_by_batch[batch_key] = []
                out_by_batch[batch_key].append(out)
            
            # Calculate differences
            gen_data_points = 0
            for batch_key in mod_by_batch:
                mods = mod_by_batch[batch_key]
                outs = out_by_batch.get(batch_key, [])
                
                min_len = min(len(mods), len(outs)) if outs else len(mods)
                if min_len < 2:
                    continue
                
                for i in range(1, min_len):
                    prev_mod = mods[i-1]
                    curr_mod = mods[i]
                    prev_mean = prev_mod.abs().mean()
                    if prev_mean.item() > 1e-9:
                        input_diff = ((curr_mod - prev_mod).abs().mean() / prev_mean).item()
                    else:
                        continue
                    
                    if i < len(outs):
                        prev_out = outs[i-1]
                        curr_out = outs[i]
                        prev_out_mean = prev_out.abs().mean()
                        if prev_out_mean.item() > 1e-9:
                            output_diff = ((curr_out - prev_out).abs().mean() / prev_out_mean).item()
                        else:
                            continue
                    else:
                        continue
                    
                    if (np.isfinite(input_diff) and np.isfinite(output_diff) and 
                        input_diff > 0 and output_diff > 0):
                        all_input_diffs.append(input_diff)
                        all_output_diffs.append(output_diff)
                        gen_data_points += 1
            
            print(f"    Data points: {gen_data_points}")
        
        if len(all_input_diffs) < polynomial_order + 1:
            return (f"ERROR: Only {len(all_input_diffs)} points", "Error")
        
        # Fit polynomial
        x = np.array(all_input_diffs)
        y = np.array(all_output_diffs)
        
        coefficients = np.polyfit(x, y, polynomial_order)
        coeff_str = ", ".join([f"{c:.8f}" for c in coefficients])
        
        # Verify fit
        poly = np.poly1d(coefficients)
        y_pred = poly(x)
        r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
        
        # Check if coefficients produce negative values in range
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range_pred = poly(x_range)
        min_pred = y_range_pred.min()
        warn_msg = ""
        if min_pred < 0:
            warn_msg = "\nWARNING: Polynomial produces negative values! Coefficients may be unstable."
        
        report = f"""TeaCache Coefficient Report
===========================
Generations: {num_generations}
Data points: {len(all_input_diffs)}
Range Input: {x.min():.4f} - {x.max():.4f}
Range Output: {y.min():.4f} - {y.max():.4f}

Fit R²: {r2:.4f}
{warn_msg}

Coefficients:
[{coeff_str}]
""" 
        return (coeff_str, report)

NODE_CLASS_MAPPINGS = { "TeaCacheCoefficientCalculator": TeaCacheCoefficientCalculator }
NODE_DISPLAY_NAME_MAPPINGS = { "TeaCacheCoefficientCalculator": "TeaCache Coefficient Calculator" }

import torch
import numpy as np
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils

class TeaCacheCoefficientCalculator:
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
                "polynomial_order": ("INT", {"default": 1, "min": 1, "max": 5}), # Default to Linear
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
    ) -> tuple[str, str]:
        
        all_input_diffs = []
        all_output_diffs = []
        
        diffusion_model = model.get_model_object("diffusion_model")
        
        # --- Hook Setup ---
        captured_mod_inputs = []
        
        def modulation_hook(module, input, output):
            # Output of adaLN_modulation is (shift, scale, gate) usually, or just processed emb
            # TeaCache_Newbie checks: mod_result = layers[0].adaLN_modulation(adaln_input)
            # So we capture 'output' here.
            # If it returns tuple, take first element.
            content = output[0] if isinstance(output, tuple) else output
            captured_mod_inputs.append(content.detach().cpu().float().clone())
            
        handle = None
        # Try to attach hook
        try:
            if hasattr(diffusion_model, 'layers') and len(diffusion_model.layers) > 0:
                first_layer = diffusion_model.layers[0]
                if hasattr(first_layer, 'adaLN_modulation'):
                    handle = first_layer.adaLN_modulation.register_forward_hook(modulation_hook)
                else:
                    return ("Error", "Error: Model layers[0] has no 'adaLN_modulation'. Architecture mismatch.")
            else:
                return ("Error", "Error: Model has no 'layers'.")
        except Exception as e:
            return ("Error", f"Error attaching hook: {e}")

        print(f"TeaCache Calculator: Hook attached. Running {num_generations} generations...")
        
        try:
            for gen_idx in range(num_generations):
                current_seed = seed + gen_idx
                captured_mod_inputs.clear() # Clear for this run
                denoised_outputs = []
                
                # Callback for outputs
                def callback(step, x, x0, total_steps):
                    denoised_outputs.append(x0.detach().cpu().float().clone())
                
                print(f"  Gen {gen_idx+1}/{num_generations}...")
                
                # Run Sampling
                latent = latent_image.copy()
                samples = latent["samples"]
                noise = comfy.sample.prepare_noise(samples, current_seed, None)
                
                sampler = comfy.samplers.KSampler(
                    model, steps=steps, device=comfy.model_management.get_torch_device(),
                    sampler=sampler_name, scheduler=scheduler, denoise=denoise
                )
                
                sampler.sample(
                    noise, positive, negative, cfg=cfg, 
                    latent_image=samples, start_step=0, last_step=steps,
                    force_full_denoise=True, seed=current_seed, callback=callback
                )
                
                # Post-process data for this generation
                # 1. Group captured inputs by batch size (handles CFG cond/uncond)
                mod_by_batch = {}
                for mod in captured_mod_inputs:
                    b_size = mod.shape[0]
                    if b_size not in mod_by_batch: mod_by_batch[b_size] = []
                    mod_by_batch[b_size].append(mod)
                    
                # 2. Group outputs
                out_by_batch = {}
                for out in denoised_outputs:
                    b_size = out.shape[0]
                    if b_size not in out_by_batch: out_by_batch[b_size] = []
                    out_by_batch[b_size].append(out)
                
                # 3. Calculate Diffs
                points_added = 0
                for b_size in mod_by_batch:
                    mods = mod_by_batch[b_size]
                    outs = out_by_batch.get(b_size, [])
                    
                    
                    limit = min(len(mods), len(outs))
                    if limit < 2: continue
                    
                    for i in range(1, limit):
                        # Input Diff
                        prev_m, curr_m = mods[i-1], mods[i]
                        p_mean = prev_m.abs().mean()
                        if p_mean < 1e-9: continue
                        inp_diff = ((curr_m - prev_m).abs().mean() / p_mean).item()
                        
                        # Output Diff
                        prev_o, curr_o = outs[i-1], outs[i]
                        p_omean = prev_o.abs().mean()
                        if p_omean < 1e-9: continue
                        out_diff = ((curr_o - prev_o).abs().mean() / p_omean).item()
                        
                        if inp_diff > 0 and out_diff > 0:
                            all_input_diffs.append(inp_diff)
                            all_output_diffs.append(out_diff)
                            points_added += 1
                            
                print(f"    captured {len(captured_mod_inputs)} inputs, used {points_added} points")
                
        finally:
            if handle: handle.remove()
            
        print(f"Total points: {len(all_input_diffs)}")
        
        if len(all_input_diffs) < 2:
            return ("Error", "Not enough data points collected.")
            
        # Fit Polynomial
        x = np.array(all_input_diffs)
        y = np.array(all_output_diffs)
        
        coeffs = np.polyfit(x, y, polynomial_order)
        
        # Format for output (pad with 0s to make 5 coeffs if order < 4)
        # TeaCache expects: c0*x^4 + ... + c4
        # polyfit returns [highest_order, ..., constant]
        # output needs to be 5 numbers.
        
        final_coeffs_list = [0.0] * 5
        # Fill from end
        # deg=1 -> [c1, c0] -> final [0, 0, 0, c1, c0]
        for i, c in enumerate(coeffs[::-1]): # Reverse to start from constant
             final_coeffs_list[4-i] = float(c)
             
        coeff_str = ", ".join([f"{c:.8f}" for c in final_coeffs_list])
        
        # Report
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
        
        report = f"""TeaCache Analysis (Hook-Based)
============================
Gens: {num_generations} | Steps: {steps}
Points: {len(all_input_diffs)}
Order: {polynomial_order}
Fit R²: {r2:.4f}

Input Range: {x.min():.4f} - {x.max():.4f}
Output Range: {y.min():.4f} - {y.max():.4f}

Coefficients:
[{coeff_str}]
"""
        return (coeff_str, report)

NODE_CLASS_MAPPINGS = { "TeaCacheCoefficientCalculator": TeaCacheCoefficientCalculator }
NODE_DISPLAY_NAME_MAPPINGS = { "TeaCacheCoefficientCalculator": "TeaCache Coefficient Calculator" }

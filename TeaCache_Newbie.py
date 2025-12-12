"""
TeaCache for Newbie/Lumina2 Models

This is a modified version of TeaCache adapted for the Newbie model architecture.
The main differences from standard Lumina2 TeaCache:
1. TimestepEmbedder.forward() doesn't accept dtype argument
2. Different forward signature (no num_tokens parameter)
3. patchify_and_embed uses adaln_input instead of num_tokens
"""

import torch
import numpy as np
from comfy.ldm.common_dit import pad_to_patch_size
from unittest.mock import patch
import re

DEFAULT_COEFFICIENTS = [0, 0, 0, 4.11423217, 0.36885889]


def teacache_forward_newbie(
    self, x, t, cap_feats, cap_mask, **kwargs
):
    """
    TeaCache-enabled forward pass for Newbie models.
    
    Signature matches the original NextDiT_CLIP.forward exactly:
    forward(self, x, t, cap_feats, cap_mask, **kwargs)
    """
    # Get TeaCache options from self (stored by wrapper) since transformer_options isn't propagated
    teacache_opts = getattr(self, '_teacache_current_options', {})
    
    if not hasattr(self, 'teacache_state'):
        self.teacache_state = {
            "cnt": 0,
            "num_steps": teacache_opts.get("num_steps"),
            "cache": teacache_opts.get("cache", {}),
            "uncond_seq_len": teacache_opts.get("uncond_seq_len")
        }
    if not isinstance(self.teacache_state.get("cache"), dict):
        self.teacache_state["cache"] = {}

    if self.teacache_state.get("num_steps") is None and teacache_opts.get("num_steps") is not None:
        self.teacache_state["num_steps"] = teacache_opts.get("num_steps")

    enable_teacache = teacache_opts.get('enable_teacache', False)
    
    # Store original dimensions for cropping
    bs, c_channels, h_img, w_img = x.shape
    
    # Handle CLIP embeddings
    clip_text_pooled = kwargs.get('clip_text_pooled')
    clip_img_pooled = kwargs.get('clip_img_pooled')
    
    # Pad to patch size (same as original)
    x = pad_to_patch_size(x, (self.patch_size, self.patch_size))
    
    # === EXACT SAME FLOW AS NextDiT_CLIP.forward ===
    # Note: t is already the timestep value, no conversion needed (ComfyUI handles this)
    
    # Step 1: Embed timestep
    t_emb = self.t_embedder(t)
    adaln_input = t_emb
    
    # cap_feats and cap_mask are already passed correctly from ComfyUI
    
    # Embed cap_feats (this is what the original model does)
    if hasattr(self, 'cap_embedder') and cap_feats is not None:

        cap_feats = self.cap_embedder(cap_feats)
    
    # Handle CLIP text pooled
    if hasattr(self, 'clip_text_pooled_proj') and clip_text_pooled is not None:
        clip_emb = self.clip_text_pooled_proj(clip_text_pooled)
        combined_features = torch.cat([t_emb, clip_emb], dim=-1)
        if hasattr(self, 'time_text_embed'):
            adaln_input = self.time_text_embed(combined_features)
    
    # Handle CLIP image pooled
    if hasattr(self, 'clip_img_pooled_embedder') and clip_img_pooled is not None:
        clip_img_pooled_emb = self.clip_img_pooled_embedder(clip_img_pooled)
        adaln_input = adaln_input + clip_img_pooled_emb
    
    # Step 3: Patchify and embed
    x_is_tensor = isinstance(x, torch.Tensor)
    x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, adaln_input)
    freqs_cis = freqs_cis.to(x.device)
    
    # === TEACACHE LOGIC ===
    max_seq_len = x.shape[1]
    should_calc = True
    # enable_teacache already set from teacache_opts above
    current_cache = None
    modulated_inp = None
    
    if enable_teacache:
        cache_key = max_seq_len
        if cache_key not in self.teacache_state['cache']:
            self.teacache_state['cache'][cache_key] = {
                "accumulated_rel_l1_distance": 0.0,
                "previous_modulated_input": None,
                "previous_residual": None,
            }
        current_cache = self.teacache_state['cache'][cache_key]
        
        try:
            if self.layers and hasattr(self.layers[0], 'adaLN_modulation'):
                mod_result = self.layers[0].adaLN_modulation(adaln_input.clone())
                if isinstance(mod_result, (list, tuple)) and len(mod_result) > 0:
                    modulated_inp = mod_result[0]
                elif torch.is_tensor(mod_result):
                    modulated_inp = mod_result
                else:
                    raise ValueError("adaLN_modulation returned unexpected type")
            else:
                raise AttributeError("Layer 0 or adaLN_modulation not found")
        except Exception as e:
            print(f"Warning: TeaCache - Failed to get modulated_inp: {e}. Disabling cache for this step.")
            enable_teacache = False
            should_calc = True
            modulated_inp = None
            if current_cache:
                current_cache["previous_modulated_input"] = None
                current_cache["accumulated_rel_l1_distance"] = 0.0

    if enable_teacache and modulated_inp is not None and current_cache is not None:
        num_steps_in_state = self.teacache_state.get("num_steps")
        if num_steps_in_state is None or num_steps_in_state == 0:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        elif self.teacache_state['cnt'] == 0 or self.teacache_state['cnt'] == num_steps_in_state - 1:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            if current_cache.get("previous_modulated_input") is not None:
                coefficients = teacache_opts.get('coefficients', DEFAULT_COEFFICIENTS)
                if not isinstance(coefficients, (list, tuple)):
                    coefficients = DEFAULT_COEFFICIENTS
                
                try:
                    rescale_func = np.poly1d(coefficients)
                except Exception as e:
                    print(f"Warning: TeaCache np.poly1d failed: {e}. Using default.")
                    rescale_func = np.poly1d(DEFAULT_COEFFICIENTS)

                prev_mod_input = current_cache["previous_modulated_input"]
                if prev_mod_input.shape != modulated_inp.shape:
                    should_calc = True
                    current_cache["accumulated_rel_l1_distance"] = 0.0
                else:
                    prev_mean = prev_mod_input.abs().mean()
                    if prev_mean.item() > 1e-9:
                        rel_l1_change = ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).cpu().item()
                    else:
                        rel_l1_change = 0.0 if modulated_inp.abs().mean().item() < 1e-9 else float('inf')
                    
                    rescaled_value = rescale_func(rel_l1_change)
                    if np.isnan(rescaled_value) or np.isinf(rescaled_value):
                        current_cache["accumulated_rel_l1_distance"] = float('inf')
                    else:
                        current_cache["accumulated_rel_l1_distance"] += rescaled_value

                    if current_cache["accumulated_rel_l1_distance"] < teacache_opts.get('rel_l1_thresh', 0.3):

                        should_calc = False
                    else:
                        should_calc = True
                        current_cache["accumulated_rel_l1_distance"] = 0.0
            else:
                should_calc = True
                current_cache["accumulated_rel_l1_distance"] = 0.0

        current_cache["previous_modulated_input"] = modulated_inp.clone()

        if self.teacache_state.get('uncond_seq_len') is None:
            self.teacache_state['uncond_seq_len'] = cache_key

        if num_steps_in_state is not None and cache_key != self.teacache_state.get('uncond_seq_len'):
            self.teacache_state['cnt'] += 1
            if self.teacache_state['cnt'] >= num_steps_in_state:
                self.teacache_state['cnt'] = 0

    # === PROCESS THROUGH LAYERS ===
    can_reuse_residual = (enable_teacache and 
                         not should_calc and 
                         current_cache and 
                         current_cache.get("previous_residual") is not None and
                         current_cache["previous_residual"].shape == x.shape)

    if can_reuse_residual:
        processed_x = x + current_cache["previous_residual"]


    else:
        original_x = x.clone()
        current_x = x
        for layer in self.layers:
            current_x = layer(current_x, mask, freqs_cis, adaln_input)

        if enable_teacache and current_cache:
            current_cache["previous_residual"] = current_x - original_x
            current_cache["accumulated_rel_l1_distance"] = 0.0
        processed_x = current_x

    # Step 4: Final layer and unpatchify (same as original)
    output = self.final_layer(processed_x, adaln_input)
    output = self.unpatchify(output, img_size, cap_size, return_tensor=x_is_tensor)
    
    # Crop to original size (original model doesn't crop, but we padded earlier)
    if isinstance(output, torch.Tensor):
        output = output[:, :, :h_img, :w_img]


    return output


class TeaCache_Newbie:
    """
    TeaCache node for Newbie/modified Lumina2 models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.6, "min": 0.0, "step": 0.001}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "The start percentage of the steps that will apply TeaCache."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "The end percentage of the steps that will apply TeaCache."}),
                "coefficients_string": ("STRING", {
                    "multiline": True, 
                    "default": str(DEFAULT_COEFFICIENTS),
                    "tooltip": "Coefficients for np.poly1d. Format: 393.7, -603.5, 209.1, -23.0, 0.86"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_teacache"
    CATEGORY = "teacache"

    def patch_teacache(self, model, rel_l1_thresh, start_percent, end_percent, coefficients_string):
        if rel_l1_thresh == 0:
            try:
                diffusion_model = model.get_model_object("diffusion_model")
                if hasattr(diffusion_model, 'teacache_state'):
                    delattr(diffusion_model, 'teacache_state')
            except:
                pass
            return (model,)

        parsed_coefficients = DEFAULT_COEFFICIENTS
        try:
            s = re.sub(r'[\[\]\s]', '', coefficients_string.strip())
            if s:
                coeff_list = [float(item) for item in s.split(',') if item]
                if coeff_list:
                    parsed_coefficients = coeff_list
        except Exception as e:
            print(f"Warning: TeaCache - Error parsing coefficients: {e}. Using defaults.")

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}

        new_model.model_options["transformer_options"]["cache"] = {}
        new_model.model_options["transformer_options"]["uncond_seq_len"] = None
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        new_model.model_options["transformer_options"]["coefficients"] = parsed_coefficients
        
        diffusion_model = new_model.get_model_object("diffusion_model")

        if hasattr(diffusion_model, 'teacache_state'):
            delattr(diffusion_model, 'teacache_state')

        context_patch_manager = patch.multiple(
            diffusion_model,
            forward=teacache_forward_newbie.__get__(diffusion_model, diffusion_model.__class__)
        )

        def unet_wrapper_function(model_function, kwargs):
            input_val = kwargs["input"]
            timestep = kwargs["timestep"]
            c_condition_dict = kwargs["c"]

            if not isinstance(c_condition_dict, dict): 
                c_condition_dict = {}
            if "transformer_options" not in c_condition_dict or not isinstance(c_condition_dict["transformer_options"], dict):
                c_condition_dict["transformer_options"] = {}

            for key, value in new_model.model_options["transformer_options"].items():
                if key not in c_condition_dict["transformer_options"]:
                    c_condition_dict["transformer_options"][key] = value

            current_step_index = 0
            if "sample_sigmas" not in c_condition_dict["transformer_options"] or \
               c_condition_dict["transformer_options"]["sample_sigmas"] is None:
                print("warning: TeaCache - 'sample_sigmas' not found. TeaCache might not work correctly.")
                c_condition_dict["transformer_options"]["enable_teacache"] = False
                c_condition_dict["transformer_options"]["num_steps"] = 1
                if hasattr(diffusion_model, 'teacache_state'):
                    delattr(diffusion_model, 'teacache_state')
            else:
                sigmas = c_condition_dict["transformer_options"]["sample_sigmas"]
                total_sampler_steps = len(sigmas)
                if total_sampler_steps > 0:
                    c_condition_dict["transformer_options"]["num_steps"] = total_sampler_steps
                else:
                    c_condition_dict["transformer_options"]["num_steps"] = 1
                    c_condition_dict["transformer_options"]["enable_teacache"] = False

                if hasattr(diffusion_model, 'teacache_state') and diffusion_model.teacache_state is not None:
                    if diffusion_model.teacache_state.get("num_steps") != total_sampler_steps and total_sampler_steps > 0:
                        delattr(diffusion_model, 'teacache_state')
                        c_condition_dict["transformer_options"]["cache"] = {}

                current_timestep = timestep[0].to(device=sigmas.device, dtype=sigmas.dtype)
                close_mask = torch.isclose(sigmas, current_timestep, atol=1e-6)
                if close_mask.any():
                    matched_step_index = torch.nonzero(close_mask, as_tuple=True)[0]
                    current_step_index = matched_step_index[0].item()
                else:
                    current_step_index = 0 
                    if total_sampler_steps > 1:
                        try:
                            indices = torch.where(sigmas >= current_timestep)[0]
                            if len(indices) > 0:
                                current_step_index = indices[-1].item()
                            else:
                                current_step_index = total_sampler_steps - 1
                        except:
                            pass

                current_percent = 0.0
                if total_sampler_steps > 1:
                    current_percent = current_step_index / max(1, (total_sampler_steps - 1))
                elif total_sampler_steps <= 0:
                    c_condition_dict["transformer_options"]["enable_teacache"] = False

                if start_percent <= current_percent <= end_percent and total_sampler_steps > 0:
                    c_condition_dict["transformer_options"]["enable_teacache"] = True
                else:
                    c_condition_dict["transformer_options"]["enable_teacache"] = False

            if current_step_index == 0:
                # Only reset ONCE at start of generation, not on every CFG call
                # Track using a generation marker based on sigmas tensor id
                current_gen_id = id(sigmas)
                last_gen_id = getattr(diffusion_model, '_teacache_generation_id', None)
                if last_gen_id != current_gen_id:
                    if hasattr(diffusion_model, 'teacache_state'):
                        delattr(diffusion_model, 'teacache_state')
                    c_condition_dict["transformer_options"]["cache"] = {}
                    diffusion_model._teacache_generation_id = current_gen_id

            # Store current options on diffusion_model so forward can access them
            diffusion_model._teacache_current_options = dict(c_condition_dict["transformer_options"])

            with context_patch_manager:
                return model_function(input_val, timestep, **c_condition_dict)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)



# Node registration
NODE_CLASS_MAPPINGS = {
    "TeaCache_Newbie": TeaCache_Newbie,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TeaCache_Newbie": "TeaCache (Newbie)",
}

"""
TeaCache for Newbie/Lumina2 Models

Adapted for the updated Newbie code
"""

import torch
import numpy as np
from comfy.ldm.common_dit import pad_to_patch_size
from unittest.mock import patch
import re

DEFAULT_COEFFICIENTS = [0, 0, 0, 4.11423217, 0.36885889]

def _get_clip_from_kwargs(transformer_options: dict, kwargs: dict, key: str):
    if key in kwargs:
        return kwargs.get(key)
    if transformer_options is not None and key in transformer_options:
        return transformer_options.get(key)
    extra = transformer_options.get("extra_cond", None) if transformer_options else None
    if isinstance(extra, dict) and key in extra:
        return extra.get(key)
    return None

def teacache_forward_newbie(
    self, 
    x: torch.Tensor, 
    timesteps: torch.Tensor, 
    context: torch.Tensor, 
    num_tokens: int = None,
    attention_mask=None, 
    transformer_options: dict = {}, 
    **kwargs
):
    """
    TeaCache-enabled forward pass matching new NewbieNextDiT._forward signature.
    """
    # Get TeaCache options: Prefer passed options, fall back to stored
    teacache_opts = transformer_options if transformer_options and 'enable_teacache' in transformer_options else getattr(self, '_teacache_current_options', {})
    
    # Initialize state if needed
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
    
    # Check start/end logic
    start_step = teacache_opts.get("teacache_start_step", 0)
    end_step = teacache_opts.get("teacache_end_step", 10000)
    current_cnt = self.teacache_state['cnt']

    if current_cnt < start_step or current_cnt >= end_step:
        enable_teacache = False
    
    # === MODEL LOGIC START (Matches newbie/model.py) ===

    t = timesteps
    cap_feats = context
    cap_mask = attention_mask
    bs, c, h, w = x.shape
    
    x = pad_to_patch_size(x, (self.patch_size, self.patch_size))
    
    t_emb = self.t_embedder(t, dtype=x.dtype)
    adaln_input = t_emb
    
    clip_text_pooled = _get_clip_from_kwargs(transformer_options, kwargs, "clip_text_pooled")
    clip_img_pooled = _get_clip_from_kwargs(transformer_options, kwargs, "clip_img_pooled")
    
    if clip_text_pooled is not None:
        if clip_text_pooled.dim() > 2:
            clip_text_pooled = clip_text_pooled.view(clip_text_pooled.shape[0], -1)
        clip_text_pooled = clip_text_pooled.to(device=t_emb.device, dtype=t_emb.dtype)
        
        if hasattr(self, 'clip_text_pooled_proj'):
            clip_emb = self.clip_text_pooled_proj(clip_text_pooled)
            if hasattr(self, 'time_text_embed'):
                adaln_input = self.time_text_embed(torch.cat([t_emb, clip_emb], dim=-1))

    if clip_img_pooled is not None:
        if clip_img_pooled.dim() > 2:
            clip_img_pooled = clip_img_pooled.view(clip_img_pooled.shape[0], -1)
        clip_img_pooled = clip_img_pooled.to(device=t_emb.device, dtype=t_emb.dtype)
        if hasattr(self, 'clip_img_pooled_embedder'):
            adaln_input = adaln_input + self.clip_img_pooled_embedder(clip_img_pooled)
            
    if isinstance(cap_feats, torch.Tensor):
        try:
            target_dtype = next(self.cap_embedder.parameters()).dtype
        except StopIteration:
            target_dtype = cap_feats.dtype
        cap_feats = cap_feats.to(device=t_emb.device, dtype=target_dtype)
        
    if hasattr(self, 'cap_embedder'):
        cap_feats = self.cap_embedder(cap_feats)
        
    # TeaCache Logic Setup
    should_calc = True
    current_cache = None
    modulated_inp = None

    if enable_teacache:
        try:
             if self.layers and hasattr(self.layers[0], 'adaLN_modulation'):
                mod_result = self.layers[0].adaLN_modulation(adaln_input.clone())
                if isinstance(mod_result, (list, tuple)) and len(mod_result) > 0:
                    modulated_inp = mod_result[0]
                elif torch.is_tensor(mod_result):
                    modulated_inp = mod_result
        except Exception:
            enable_teacache = False

        if enable_teacache and modulated_inp is not None:
             # Create Independent Caches for different shapes (Cond vs Uncond)
             # Use modulated_inp.shape AND num_tokens as key to differentiate.
             # adaln_input shape alone is not unique enough (batch size same).
             cache_key_shape = f"{modulated_inp.shape}_{num_tokens}"
            
             if "cache" not in transformer_options:
                 transformer_options["cache"] = {}
            
             # Root cache
             root_cache = transformer_options["cache"]
            
             # Sub cache for this specific shape
             if cache_key_shape not in root_cache:
                 root_cache[cache_key_shape] = {
                     "previous_modulated_input": None,
                     "accumulated_rel_l1_distance": 0.0,
                     "previous_residual": None
                 }
             current_cache = root_cache[cache_key_shape]

    
    # TeaCache Decision Logic
    # No need to check 'modulated_inp is not None' as it is adaln_input which is required.
    if enable_teacache and current_cache is not None:
        num_steps_in_state = self.teacache_state.get("num_steps")
        
        # Heuristic: if first step or last step, always calc
        if num_steps_in_state is None or num_steps_in_state == 0:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        elif self.teacache_state['cnt'] == 0 or self.teacache_state['cnt'] == num_steps_in_state - 1:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            if current_cache.get("previous_modulated_input") is not None:
                coefficients = teacache_opts.get('coefficients', DEFAULT_COEFFICIENTS)
                try:
                    rescale_func = np.poly1d(coefficients)
                except:
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
                        rel_l1_change = 0.0
                    
                    rescaled_value = rescale_func(rel_l1_change)
                    if np.isnan(rescaled_value) or np.isinf(rescaled_value):
                         current_cache["accumulated_rel_l1_distance"] = float('inf')
                    else:
                         current_cache["accumulated_rel_l1_distance"] += rescaled_value

                    thresh = teacache_opts.get('rel_l1_thresh', 0.3)
                    if current_cache["accumulated_rel_l1_distance"] < thresh:
                        should_calc = False
                        # print(f"CACHE HIT! Diff: {rel_l1_change:.4f}, Rescaled: {rescaled_value:.4f}, Accum: {current_cache['accumulated_rel_l1_distance']:.4f} < {thresh}")
                    else:
                        should_calc = True
                        # print(f"CALC: Accum {current_cache['accumulated_rel_l1_distance']:.4f} >= {thresh}")
                        current_cache["accumulated_rel_l1_distance"] = 0.0
            else:
                 should_calc = True
                 current_cache["accumulated_rel_l1_distance"] = 0.0
    else:
        if not enable_teacache: 
            # print("TeaCache DISABLED") # Commented out to reduce spam
            pass
        elif modulated_inp is None: 
            # print("Modulated Input NONE") # Commented out
            pass
    
    if current_cache is not None and modulated_inp is not None:
        current_cache["previous_modulated_input"] = modulated_inp.clone()

        
        # Update Step Count based on Timestep change
        current_t_val = t[0].item() if t.numel() > 0 else 0
        last_t_val = self.teacache_state.get('last_timestep')
        
        if last_t_val is not None and abs(current_t_val - last_t_val) > 1e-4:
            self.teacache_state['cnt'] += 1
            if self.teacache_state['cnt'] >= num_steps_in_state:
                 self.teacache_state['cnt'] = num_steps_in_state - 1 # Clamp

        self.teacache_state['last_timestep'] = current_t_val


    patches = transformer_options.get("patches", {})
    x_is_tensor = True
    
    img, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
        x, cap_feats, cap_mask, adaln_input, num_tokens, transformer_options=transformer_options
    )
    freqs_cis = freqs_cis.to(img.device)
    
    can_reuse_residual = (enable_teacache and 
                         not should_calc and 
                         current_cache and 
                         current_cache.get("previous_residual") is not None and
                         current_cache["previous_residual"].shape == img.shape)
    
    if can_reuse_residual:
        img = img + current_cache["previous_residual"]
    else:
        original_img = img.clone()

        for i, layer in enumerate(self.layers):
            img = layer(img, mask, freqs_cis, adaln_input, transformer_options=transformer_options)
            if "double_block" in patches:
                for p in patches["double_block"]:
                     out = p({
                        "img": img[:, cap_size[0] :],
                        "txt": img[:, : cap_size[0]],
                        "pe": freqs_cis[:, cap_size[0] :],
                        "vec": adaln_input,
                        "x": x,
                        "block_index": i,
                        "transformer_options": transformer_options,
                     })
                     if isinstance(out, dict):
                        if "img" in out: img[:, cap_size[0] :] = out["img"]
                        if "txt" in out: img[:, : cap_size[0]] = out["txt"]
        
        if enable_teacache and current_cache:
            current_cache["previous_residual"] = img - original_img
            # current_cache["accumulated_rel_l1_distance"] = 0.0 # This was WRONG. We reset only when we DECIDE to calc, which we did earlier.

            
    img = self.final_layer(img, adaln_input)
    img = self.unpatchify(img, img_size, cap_size, return_tensor=x_is_tensor)
    img = img[:, :, :h, :w]
    return img


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
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "coefficients_string": ("STRING", {
                    "multiline": True, 
                    "default": str(DEFAULT_COEFFICIENTS)
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
            print(f"Warning: TeaCache - using default coeffs: {e}")

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}

        new_model.model_options["transformer_options"]["cache"] = {}
        new_model.model_options["transformer_options"]["uncond_seq_len"] = None
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        new_model.model_options["transformer_options"]["coefficients"] = parsed_coefficients
        
        diffusion_model = new_model.get_model_object("diffusion_model")
        
        # Cleanup old state
        if hasattr(diffusion_model, 'teacache_state'):
            delattr(diffusion_model, 'teacache_state')
        if hasattr(diffusion_model, '_teacache_generation_id'):
            delattr(diffusion_model, '_teacache_generation_id')

        # Patch forward (_forward is the method implemented in NewbieNextDiT)
        context_patch_manager = patch.multiple(
            diffusion_model,
            _forward=teacache_forward_newbie.__get__(diffusion_model, diffusion_model.__class__)
        ).start()

        def unet_wrapper_function(model_function, kwargs):
            input_val = kwargs["input"]
            timestep = kwargs["timestep"]
            c_condition_dict = kwargs["c"]
            # Ensure transformer_options exists
            if "transformer_options" not in c_condition_dict:
                c_condition_dict["transformer_options"] = {}

            # Propagate options
            opts = new_model.model_options["transformer_options"]
            c_condition_dict["transformer_options"].update({
                "rel_l1_thresh": opts["rel_l1_thresh"],
                "enable_teacache": True,
                "coefficients": opts["coefficients"],
                "cache": opts["cache"],
                "uncond_seq_len": opts["uncond_seq_len"]
            })
            
            # Attach options to the diffusion_model instance so 'self' can access them
            # model_function is usually apply_model, so __self__ is the Model object (e.g. BaseModel)
            # The actual UNet is in .diffusion_model
            model_obj = model_function.__self__ if hasattr(model_function, '__self__') else model_function
            if hasattr(model_obj, "diffusion_model"):
                target_model = model_obj.diffusion_model
            else:
                target_model = model_obj # Fallback
            
            # DEBUG PRINT
            # print(f"TeaCache Wrapper: Set Enable=True. ID: {id(target_model)}")

            sigmas = c_condition_dict["transformer_options"].get("sample_sigmas")
            if sigmas is not None:
                total_steps = len(sigmas) - 1
                c_condition_dict["transformer_options"]["num_steps"] = total_steps
                
                # Convert percent to step indices
                start_step = int(total_steps * start_percent)
                end_step = int(total_steps * end_percent)
                c_condition_dict["transformer_options"]["teacache_start_step"] = start_step
                c_condition_dict["transformer_options"]["teacache_end_step"] = end_step
            
            current_gen_id = id(sigmas) if sigmas is not None else 0
            last_gen_id = getattr(target_model, '_teacache_generation_id', None)
            
            if last_gen_id != current_gen_id:
                if hasattr(target_model, 'teacache_state'):
                    delattr(target_model, 'teacache_state')
                c_condition_dict["transformer_options"]["cache"] = {} 
                target_model._teacache_generation_id = current_gen_id
            
            target_model._teacache_current_options = c_condition_dict["transformer_options"]
            
            return model_function(input_val, timestep, **c_condition_dict)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)

NODE_CLASS_MAPPINGS = { "TeaCache_Newbie": TeaCache_Newbie }
NODE_DISPLAY_NAME_MAPPINGS = { "TeaCache_Newbie": "TeaCache (Newbie)" }

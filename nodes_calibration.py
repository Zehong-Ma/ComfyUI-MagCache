import math
import torch
import comfy.ldm.common_dit
import comfy.model_management as mm
import numpy as np

from torch import Tensor
from einops import repeat
from typing import Optional
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding, apply_mod
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.ldm.wan.model import sinusoidal_embedding_1d


def magcache_flux_calibration_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        total_infer_steps = transformer_options.get("total_infer_steps")
    
        if not hasattr(self, 'calibration_data'):
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = None
        
        
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y[:,:self.params.vec_in_dim])
        txt = self.txt_in(txt)

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        blocks_replace = patches_replace.get("dit", {})

        ori_img = img.clone()
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"],
                                                txt=args["txt"],
                                                vec=args["vec"],
                                                pe=args["pe"],
                                                attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("double_block", i)]({"img": img,
                                                        "txt": txt,
                                                        "vec": vec,
                                                        "pe": pe,
                                                        "attn_mask": attn_mask},
                                                        {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img,
                                txt=txt,
                                vec=vec,
                                pe=pe,
                                attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            # PuLID attention
            if getattr(self, "pulid_data", {}):
                if i % self.pulid_double_interval == 0:
                    # Will calculate influence of all pulid nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps)
                                    & (timesteps >= node_data['sigma_end'])):
                            img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                    ca_idx += 1

        img = torch.cat((txt, img), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"],
                                    vec=args["vec"],
                                    pe=args["pe"],
                                    attn_mask=args.get("attn_mask"))
                    return out

                out = blocks_replace[("single_block", i)]({"img": img,
                                                        "vec": vec,
                                                        "pe": pe,
                                                        "attn_mask": attn_mask}, 
                                                        {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1] :, ...] += add

            # PuLID attention
            if getattr(self, "pulid_data", {}):
                real_img, txt = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    # Will calculate influence of all nodes at once
                    for _, node_data in self.pulid_data.items():
                        if torch.any((node_data['sigma_start'] >= timesteps)
                                    & (timesteps >= node_data['sigma_end'])):
                            real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                    ca_idx += 1
                img = torch.cat((txt, real_img), 1)

        img = img[:, txt.shape[1] :, ...]
        cur_residual = img - ori_img
        if self.calibration_data['step_count'] >= 1:
            # Calculate calibration metrics
            norm_ratio = (cur_residual.norm(dim=-1) / self.previous_residual.norm(dim=-1)).mean().item()
            norm_std = (cur_residual.norm(dim=-1) / self.previous_residual.norm(dim=-1)).std().item()
            cos_dist = (1 - torch.nn.functional.cosine_similarity(cur_residual, self.previous_residual, dim=-1, eps=1e-8)).mean().item()
            
            # Store metrics
            self.calibration_data['norm_ratios'].append(round(norm_ratio, 5))
            self.calibration_data['norm_stds'].append(round(norm_std, 5))
            self.calibration_data['cos_dists'].append(round(cos_dist, 5))
            if self.calibration_data['step_count'] >= (total_infer_steps-1):
                print("mag_ratios")
                print(self.calibration_data['norm_ratios'])
                print("mag_ratio_std")
                print(self.calibration_data['norm_stds'])
                print("mag_cos_dist")
                print(self.calibration_data['cos_dists'])
        self.previous_residual = cur_residual
        self.calibration_data['step_count'] += 1  
        if total_infer_steps == self.calibration_data['step_count']: # del cache when calibration multiple times
            del self.calibration_data
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = None
            
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return img

def magcache_qwen_image_calibration_forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        control=None,
        **kwargs
    ):
        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask

        hidden_states, img_ids, orig_shape = self.process_img(x)
        num_embeds = hidden_states.shape[1]

        if ref_latents is not None:
            h = 0
            w = 0
            index = 0
            index_ref_method = kwargs.get("ref_latents_method", "index") == "index"
            for ref in ref_latents:
                if index_ref_method:
                    index += 1
                    h_offset = 0
                    w_offset = 0
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
                hidden_states = torch.cat([hidden_states, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)

        txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2, ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
        txt_ids = torch.arange(txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids).to(x.dtype).contiguous()
        del ids, txt_ids, img_ids

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        patches_replace = transformer_options.get("patches_replace", {})
        patches = transformer_options.get("patches", {})
        blocks_replace = patches_replace.get("dit", {})
        
        total_infer_steps = transformer_options.get("total_infer_steps")
    
        if not hasattr(self, 'calibration_data'):
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        if total_infer_steps*2 == self.calibration_data['step_count']: # del cache when calibration multiple times
            del self.calibration_data
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        
        ori_hidden_states = hidden_states.clone()
        for i, block in enumerate(self.transformer_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["txt"], out["img"] = block(hidden_states=args["img"], encoder_hidden_states=args["txt"], encoder_hidden_states_mask=encoder_hidden_states_mask, temb=args["vec"], image_rotary_emb=args["pe"], transformer_options=args["transformer_options"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": hidden_states, "txt": encoder_hidden_states, "vec": temb, "pe": image_rotary_emb, "transformer_options": transformer_options}, {"original_block": block_wrap})
                hidden_states = out["img"]
                encoder_hidden_states = out["txt"]
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=transformer_options,
                )

            if "double_block" in patches:
                for p in patches["double_block"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x, "block_index": i, "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        hidden_states[:, :add.shape[1]] += add
        
        cur_residual = hidden_states - ori_hidden_states
        if self.calibration_data['step_count'] >= 2:
            # Calculate calibration metrics
            norm_ratio = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).mean().item()
            norm_std = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).std().item()
            cos_dist = (1 - torch.nn.functional.cosine_similarity(cur_residual, self.previous_residual[self.calibration_data["step_count"]%2], dim=-1, eps=1e-8)).mean().item()
            
            # Store metrics
            self.calibration_data['norm_ratios'].append(round(norm_ratio, 5))
            self.calibration_data['norm_stds'].append(round(norm_std, 5))
            self.calibration_data['cos_dists'].append(round(cos_dist, 5))
            if self.calibration_data['step_count'] >= (total_infer_steps*2-1):
                print("mag_ratios")
                print(self.calibration_data['norm_ratios'])
                print("mag_ratio_std")
                print(self.calibration_data['norm_stds'])
                print("mag_cos_dist")
                print(self.calibration_data['cos_dists'])
        self.previous_residual[self.calibration_data["step_count"]%2] = cur_residual.detach()
        self.calibration_data['step_count'] += 1    
        print(self.calibration_data['step_count'])
        
        
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]

def magcache_hunyuan15_calibration_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor = None,
        txt_byt5=None,
        clip_fea=None,
        guidance: Tensor = None,
        guiding_frame_index=None,
        ref_latent=None,
        disable_time_r=False,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        total_infer_steps = transformer_options.get("total_infer_steps")
    
        if not hasattr(self, 'calibration_data'):
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        if total_infer_steps*2 == self.calibration_data['step_count']: # del cache when calibration multiple times
            del self.calibration_data
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        
        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        if (self.time_r_in is not None) and (not disable_time_r):
            w = torch.where(transformer_options['sigmas'][0] == transformer_options['sample_sigmas'])[0]  # This most likely could be improved
            if len(w) > 0:
                timesteps_r = transformer_options['sample_sigmas'][w[0] + 1]
                timesteps_r = timesteps_r.unsqueeze(0).to(device=timesteps.device, dtype=timesteps.dtype)
                vec_r = self.time_r_in(timestep_embedding(timesteps_r, 256, time_factor=1000.0).to(img.dtype))
                vec = (vec + vec_r) / 2

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            if self.vector_in is not None:
                vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
                vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            else:
                vec = torch.cat([(token_replace_vec).unsqueeze(1), (vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            if self.vector_in is not None:
                vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask, transformer_options=transformer_options)

        if self.cond_type_embedding is not None:
            self.cond_type_embedding.to(txt.device)
            cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
            txt = txt + cond_emb.to(txt.dtype)

        if self.byt5_in is not None and txt_byt5 is not None:
            txt_byt5 = self.byt5_in(txt_byt5)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(torch.ones_like(txt_byt5[:, :, 0], device=txt_byt5.device, dtype=torch.long))
                txt_byt5 = txt_byt5 + cond_emb.to(txt_byt5.dtype)
                txt = torch.cat((txt_byt5, txt), dim=1) # byt5 first for HunyuanVideo1.5
            else:
                txt = torch.cat((txt, txt_byt5), dim=1)
            txt_byt5_ids = torch.zeros((txt_ids.shape[0], txt_byt5.shape[1], txt_ids.shape[-1]), device=txt_ids.device, dtype=txt_ids.dtype)
            txt_ids = torch.cat((txt_ids, txt_byt5_ids), dim=1)

        if clip_fea is not None:
            txt_vision_states = self.vision_in(clip_fea)
            if self.cond_type_embedding is not None:
                cond_emb = self.cond_type_embedding(2 * torch.ones_like(txt_vision_states[:, :, 0], dtype=torch.long, device=txt_vision_states.device))
                txt_vision_states = txt_vision_states + cond_emb
            txt = torch.cat((txt_vision_states.to(txt.dtype), txt), dim=1)
            extra_txt_ids = torch.zeros((txt_ids.shape[0], txt_vision_states.shape[1], txt_ids.shape[-1]), device=txt_ids.device, dtype=txt_ids.dtype)
            txt_ids = torch.cat((txt_ids, extra_txt_ids), dim=1)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})
        
        ori_img = img.clone()
        for i, block in enumerate(self.double_blocks):
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"], transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt, 'transformer_options': transformer_options}, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

        img = torch.cat((img, txt), 1)

        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"], transformer_options=args["transformer_options"])
                    return out

                out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims, 'transformer_options': transformer_options}, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims, transformer_options=transformer_options)

            if control is not None: # Controlnet
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, : img_len] += add

        img = img[:, : img_len]
        

        cur_residual = img - ori_img
        if self.calibration_data['step_count'] >= 2:
            # Calculate calibration metrics
            norm_ratio = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).mean().item()
            norm_std = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).std().item()
            cos_dist = (1 - torch.nn.functional.cosine_similarity(cur_residual, self.previous_residual[self.calibration_data["step_count"]%2], dim=-1, eps=1e-8)).mean().item()
            
            # Store metrics
            self.calibration_data['norm_ratios'].append(round(norm_ratio, 5))
            self.calibration_data['norm_stds'].append(round(norm_std, 5))
            self.calibration_data['cos_dists'].append(round(cos_dist, 5))
            if self.calibration_data['step_count'] >= (total_infer_steps*2-1):
                print("mag_ratios")
                print(self.calibration_data['norm_ratios'])
                print("mag_ratio_std")
                print(self.calibration_data['norm_stds'])
                print("mag_cos_dist")
                print(self.calibration_data['cos_dists'])
        self.previous_residual[self.calibration_data["step_count"]%2] = cur_residual.detach()
        self.calibration_data['step_count'] += 1    
        # print(self.calibration_data['step_count'])
            
        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]
        img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-len(self.patch_size):]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        if img.ndim == 8:
            img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
            img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        else:
            img = img.permute(0, 3, 1, 4, 2, 5)
            img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3])
        return img


def magcache_chroma_calibration_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        guidance: Tensor = None,
        control = None,
        transformer_options={},
        attn_mask: Tensor = None,
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        total_infer_steps = transformer_options.get("total_infer_steps")
    
        if not hasattr(self, 'calibration_data'):
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        if total_infer_steps*2 == self.calibration_data['step_count']: # del cache when calibration multiple times
            del self.calibration_data
            self.calibration_data = {
                'norm_ratios': [],
                'norm_stds': [],
                'cos_dists': [],
                'step_count': 0
            }
            self.previous_residual = [None, None]
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
            
        # running on sequences img
        img = self.img_in(img)
        
        # Chroma-specific modulation vectors setup
        mod_index_length = 344
        distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
        distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)
        modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1).to(img.device, img.dtype)
        timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device, img.dtype)
        input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1).to(img.device, img.dtype)
        mod_vectors = self.distilled_guidance_layer(input_vec)
        
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        blocks_replace = patches_replace.get("dit", {})
        
            
        ori_img = img.clone()
        for i, block in enumerate(self.double_blocks):
            if i not in self.skip_mmdit:
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"],
                                                        txt=args["txt"],
                                                        vec=args["vec"],
                                                        pe=args["pe"],
                                                        attn_mask=args.get("attn_mask"))
                        return out
                    out = blocks_replace[("double_block", i)]({"img": img,
                                                                "txt": txt,
                                                                "vec": double_mod,
                                                                "pe": pe,
                                                                "attn_mask": attn_mask},
                                                                {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img,
                                    txt=txt,
                                    vec=double_mod,
                                    pe=pe,
                                    attn_mask=attn_mask)
                if control is not None:  # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add
                            
        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            if i not in self.skip_dit:
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"],
                                            vec=args["vec"],
                                            pe=args["pe"],
                                            attn_mask=args.get("attn_mask"))
                        return out
                    out = blocks_replace[("single_block", i)]({"img": img,
                                                                "vec": single_mod,
                                                                "pe": pe,
                                                                "attn_mask": attn_mask},
                                                            {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask)
                if control is not None:  # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, txt.shape[1]:, ...] += add
                            
        img = img[:, txt.shape[1]:, ...]
       
        cur_residual = img - ori_img
        if self.calibration_data['step_count'] >= 2:
            # Calculate calibration metrics
            norm_ratio = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).mean().item()
            norm_std = (cur_residual.norm(dim=-1) / self.previous_residual[self.calibration_data["step_count"]%2].norm(dim=-1)).std().item()
            cos_dist = (1 - torch.nn.functional.cosine_similarity(cur_residual, self.previous_residual[self.calibration_data["step_count"]%2], dim=-1, eps=1e-8)).mean().item()
            
            # Store metrics
            self.calibration_data['norm_ratios'].append(round(norm_ratio, 5))
            self.calibration_data['norm_stds'].append(round(norm_std, 5))
            self.calibration_data['cos_dists'].append(round(cos_dist, 5))
            if self.calibration_data['step_count'] >= (total_infer_steps*2-1):
                print("mag_ratios")
                print(self.calibration_data['norm_ratios'])
                print("mag_ratio_std")
                print(self.calibration_data['norm_stds'])
                print("mag_cos_dist")
                print(self.calibration_data['cos_dists'])
        self.previous_residual[self.calibration_data["step_count"]%2] = cur_residual.detach()
        self.calibration_data['step_count'] += 1    
        # print(self.calibration_data['step_count'])
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)
        
        return img

class MagCacheCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the MagCache will be applied to."}),
                "model_type": (["chroma_calibration", "flux_calibration", "flux_kontext_calibration", "hunyuanvideo1.5", "qwen_image"], {"default": "chroma_calibration", "tooltip": "Supported diffusion model."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_magcache"
    CATEGORY = "MagCacheCalibration"
    TITLE = "MagCache Calibration"
    
    def apply_magcache(self, model, model_type: str):
        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        diffusion_model = new_model.get_model_object("diffusion_model")

        if "chroma_calibration" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_chroma_calibration_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "flux_calibration" in model_type or "flux_kontext_calibration" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_flux_calibration_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "qwen_image" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                _forward=magcache_qwen_image_calibration_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
            # context = patch.multiple(
            #     diffusion_model,
            #     forward_orig=magcache_qwen_image_calibration_forward.__get__(diffusion_model, diffusion_model.__class__)
            # )
            
        elif "hunyuanvideo1.5" in model_type:
            is_cfg = True # only support cfg
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_hunyuan15_calibration_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        else:
            raise ValueError(f"Unknown type {model_type}")
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            cond_or_uncond = kwargs["cond_or_uncond"]
            # referenced from https://github.com/kijai/ComfyUI-KJNodes/blob/d126b62cebee81ea14ec06ea7cd7526999cb0554/nodes/model_optimization_nodes.py#L868
            sigmas = c["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    # walk from beginning of steps until crossing the timestep
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
            
            if current_step_index == 0:
                if is_cfg:
                    # uncond first
                    if (1 in cond_or_uncond) and hasattr(diffusion_model, 'magcache_state_state'):
                        delattr(diffusion_model, 'magcache_state_state')
                else:
                    if hasattr(diffusion_model, 'accumulated_rel_l1_distance'):
                        delattr(diffusion_model, 'accumulated_rel_l1_distance')
            total_infer_steps = len(sigmas)-1

            c["transformer_options"]["total_infer_steps"] = total_infer_steps
            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)
    
def patch_optimized_module():
    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        return

    if getattr(OptimizedModule, "_patched", False):
        return

    def __getattribute__(self, name):
        if name == "_orig_mod":
            return object.__getattribute__(self, "_modules")[name]
        if name in (
            "__class__",
            "_modules",
            "state_dict",
            "load_state_dict",
            "parameters",
            "named_parameters",
            "buffers",
            "named_buffers",
            "children",
            "named_children",
            "modules",
            "named_modules",
        ):
            return getattr(object.__getattribute__(self, "_orig_mod"), name)
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        return delattr(self._orig_mod, name)

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, OptimizedModule) or issubclass(
            object.__getattribute__(instance, "__class__"), cls
        )

    OptimizedModule.__getattribute__ = __getattribute__
    OptimizedModule.__delattr__ = __delattr__
    OptimizedModule.__instancecheck__ = __instancecheck__
    OptimizedModule._patched = True

def patch_same_meta():
    try:
        from torch._inductor.fx_passes import post_grad
    except ImportError:
        return

    same_meta = getattr(post_grad, "same_meta", None)
    if same_meta is None:
        return

    if getattr(same_meta, "_patched", False):
        return

    def new_same_meta(a, b):
        try:
            return same_meta(a, b)
        except Exception:
            return False

    post_grad.same_meta = new_same_meta
    new_same_meta._patched = True


NODE_CLASS_MAPPINGS = {
    "MagCacheCalibration": MagCacheCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}

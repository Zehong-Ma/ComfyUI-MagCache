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


SUPPORTED_MODELS_MAG_RATIOS = {
    "flux": np.array([1.0]+[1.21094, 1.11719, 1.07812, 1.0625, 1.03906, 1.03125, 1.03906, 1.02344, 1.03125, 1.02344, 0.98047, 1.01562, 1.00781, 1.0, 1.00781, 1.0, 1.00781, 1.0, 1.0, 0.99609, 0.99609, 0.98047, 0.98828, 0.96484, 0.95703, 0.93359, 0.89062]),
    "flux_kontext": np.array([1.0]+[1.21875, 1.0625, 1.03125, 1.03125, 1.0, 1.00781, 1.03906, 0.98047, 1.03125, 0.96875, 1.02344, 1.0, 0.99219, 1.02344, 0.98047, 0.95703, 0.98828, 0.98047, 0.88672]),
    "chroma": np.array([1.0]*2+[1.00781, 1.01562, 1.03906, 1.03906, 1.05469, 1.05469, 1.07031, 1.07031, 1.04688, 1.04688, 1.03906, 1.03906, 1.03906, 1.03906, 1.01562, 1.01562, 1.05469, 1.05469, 0.99609, 0.99609, 1.02344, 1.02344, 1.01562, 1.01562, 0.99609, 0.99609, 1.0, 1.0, 0.99219, 0.99219, 1.00781, 1.00781, 1.00781, 1.00781, 0.97656, 0.97656, 0.98828, 0.98828, 0.97266, 0.97266, 1.0, 1.0, 0.93359, 0.93359, 0.94922, 0.94922, 0.92578, 0.92578, 1.0625, 1.0625]),
    "qwen_image": np.array([1.0]*2 +[1.6875, 1.6875, 1.25, 1.25, 1.34375, 1.33594, 1.11719, 1.11719, 1.10938, 1.10938, 1.13281, 1.13281, 1.14062, 1.13281, 1.10156, 1.10156, 1.14844, 1.14844, 1.07812, 1.07812, 1.09375, 1.09375, 1.07812, 1.07812, 1.07812, 1.07031, 1.10156, 1.10156, 1.07031, 1.07031, 1.07812, 1.07812, 1.03906, 1.03906, 1.08594, 1.08594, 1.07031, 1.07031, 1.07031, 1.07031, 1.05469, 1.05469, 1.0625, 1.0625, 1.03125, 1.03125, 1.03906, 1.03906, 1.04688, 1.04688, 1.02344, 1.02344, 1.03906, 1.03906, 1.03125, 1.03125, 1.01562, 1.01562, 1.01562, 1.01562, 0.98828, 0.98828, 0.99219, 0.99219, 0.94922, 0.94922, 0.98047, 0.98047, 0.88672, 0.89062, 0.89062, 0.89062, 0.78125, 0.78125, 0.74219, 0.73828, 0.54688, 0.54688]),
    "hunyuan_video": np.array([1.0]+[1.0754, 1.27807, 1.11596, 1.09504, 1.05188, 1.00844, 1.05779, 1.00657, 1.04142, 1.03101, 1.00679, 1.02556, 1.00908, 1.06949, 1.05438, 1.02214, 1.02321, 1.03019, 1.00779, 1.03381, 1.01886, 1.01161, 1.02968, 1.00544, 1.02822, 1.00689, 1.02119, 1.0105, 1.01044, 1.01572, 1.02972, 1.0094, 1.02368, 1.0226, 0.98965, 1.01588, 1.02146, 1.0018, 1.01687, 0.99436, 1.00283, 1.01139, 0.97122, 0.98251, 0.94513, 0.97656, 0.90943, 0.85703, 0.75456]),
    "hunyuan_video1.5": np.array([1.0]*2+[1.01172, 1.00293, 0.9873, 1.0127, 1.01465, 0.98926, 0.99805, 1.00098, 0.99512, 0.99365, 0.99512, 0.99561, 0.99316, 0.99365, 0.99365, 0.99365, 0.99316, 0.99316, 0.99268, 0.99268, 0.9917, 0.9917, 0.99023, 0.99023, 0.98828, 0.98877, 0.98633, 0.98633, 0.9834, 0.9834, 0.97949, 0.97998, 0.97363, 0.97363, 0.96436, 0.96436, 0.94824, 0.94873]),
    "hunyuan_video1.5_40steps": np.array([1.0]*2+[1.04199, 1.01953, 1.01172, 1.01855, 1.00293, 1.00586, 1.00098, 1.00195, 1.0, 1.00098, 0.99805, 0.99902, 0.99854, 0.99805, 0.99658, 0.99756, 0.99707, 0.99512, 0.99609, 0.99609, 0.99658, 0.99658, 0.99658, 0.99805, 0.99658, 0.99707, 0.99561, 0.99561, 0.99561, 0.99658, 0.99658, 0.99658, 0.99609, 0.99658, 0.99512, 0.99561, 0.99463, 0.99512, 0.99463, 0.99512, 0.99463, 0.99512, 0.99365, 0.99414, 0.99365, 0.99365, 0.99219, 0.99219, 0.99219, 0.99268, 0.99072, 0.99121, 0.99072, 0.99072, 0.98926, 0.98975, 0.9873, 0.98779, 0.98535, 0.98584, 0.9834, 0.9834, 0.97998, 0.97998, 0.97607, 0.97607, 0.96973, 0.96973, 0.96338, 0.96338, 0.9502, 0.9502, 0.93066, 0.93066, 0.896, 0.896, 0.81787, 0.81836]),
    "wan2.1_t2v_1.3B": np.array([1.0]*2+[1.0124, 1.02213, 1.00166, 1.0041, 0.99791, 1.00061, 0.99682, 0.99762, 0.99634, 0.99685, 0.99567, 0.99586, 0.99416, 0.99422, 0.99578, 0.99575, 0.9957, 0.99563, 0.99511, 0.99506, 0.99535, 0.99531, 0.99552, 0.99549, 0.99541, 0.99539, 0.9954, 0.99536, 0.99489, 0.99485, 0.99518, 0.99514, 0.99484, 0.99478, 0.99481, 0.99479, 0.99415, 0.99413, 0.99419, 0.99416, 0.99396, 0.99393, 0.99388, 0.99386, 0.99349, 0.99349, 0.99309, 0.99304, 0.9927, 0.9927, 0.99228, 0.99226, 0.99171, 0.9917, 0.99137, 0.99135, 0.99068, 0.99063, 0.99005, 0.99003, 0.98944, 0.98942, 0.98849, 0.98849, 0.98758, 0.98757, 0.98644, 0.98643, 0.98504, 0.98503, 0.9836, 0.98359, 0.98202, 0.98201, 0.97977, 0.97978, 0.97717, 0.97718, 0.9741, 0.97411, 0.97003, 0.97002, 0.96538, 0.96541, 0.9593, 0.95933, 0.95086, 0.95089, 0.94013, 0.94019, 0.92402, 0.92414, 0.90241, 0.9026, 0.86821, 0.86868, 0.81838, 0.81939]),
    "wan2.1_t2v_14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
    "wan2.1_i2v_480p_14B": np.array([1.0]*2+[0.98783, 0.98993, 0.97559, 0.97593, 0.98311, 0.98319, 0.98202, 0.98225, 0.9888, 0.98878, 0.98762, 0.98759, 0.98957, 0.98971, 0.99052, 0.99043, 0.99383, 0.99384, 0.98857, 0.9886, 0.99065, 0.99068, 0.98845, 0.98847, 0.99057, 0.99057, 0.98957, 0.98961, 0.98601, 0.9861, 0.98823, 0.98823, 0.98756, 0.98759, 0.98808, 0.98814, 0.98721, 0.98724, 0.98571, 0.98572, 0.98543, 0.98544, 0.98157, 0.98165, 0.98411, 0.98413, 0.97952, 0.97953, 0.98149, 0.9815, 0.9774, 0.97742, 0.97825, 0.97826, 0.97355, 0.97361, 0.97085, 0.97087, 0.97056, 0.97055, 0.96588, 0.96587, 0.96113, 0.96124, 0.9567, 0.95681, 0.94961, 0.94969, 0.93973, 0.93988, 0.93217, 0.93224, 0.91878, 0.91896, 0.90955, 0.90954, 0.92617, 0.92616]),
    "wan2.1_i2v_720p_14B": np.array([1.0]*2+[0.99428, 0.99498, 0.98588, 0.98621, 0.98273, 0.98281, 0.99018, 0.99023, 0.98911, 0.98917, 0.98646, 0.98652, 0.99454, 0.99456, 0.9891, 0.98909, 0.99124, 0.99127, 0.99102, 0.99103, 0.99215, 0.99212, 0.99515, 0.99515, 0.99576, 0.99572, 0.99068, 0.99072, 0.99097, 0.99097, 0.99166, 0.99169, 0.99041, 0.99042, 0.99201, 0.99198, 0.99101, 0.99101, 0.98599, 0.98603, 0.98845, 0.98844, 0.98848, 0.98851, 0.98862, 0.98857, 0.98718, 0.98719, 0.98497, 0.98497, 0.98264, 0.98263, 0.98389, 0.98393, 0.97938, 0.9794, 0.97535, 0.97536, 0.97498, 0.97499, 0.973, 0.97301, 0.96827, 0.96828, 0.96261, 0.96263, 0.95335, 0.9534, 0.94649, 0.94655, 0.93397, 0.93414, 0.91636, 0.9165, 0.89088, 0.89109, 0.8679, 0.86768]),
    "wan2.1_vace_1.3B": np.array([1.0]*2+[1.00129, 1.0019, 1.00056, 1.00053, 0.99776, 0.99746, 0.99726, 0.99789, 0.99725, 0.99785, 0.9958, 0.99625, 0.99703, 0.99728, 0.99863, 0.9988, 0.99735, 0.99731, 0.99714, 0.99707, 0.99697, 0.99687, 0.9969, 0.99683, 0.99695, 0.99702, 0.99697, 0.99701, 0.99608, 0.99617, 0.99721, 0.9973, 0.99649, 0.99657, 0.99659, 0.99667, 0.99727, 0.99731, 0.99603, 0.99612, 0.99652, 0.99659, 0.99635, 0.9964, 0.9958, 0.99585, 0.99581, 0.99585, 0.99573, 0.99579, 0.99531, 0.99534, 0.99505, 0.99508, 0.99481, 0.99484, 0.99426, 0.99433, 0.99403, 0.99406, 0.99357, 0.9936, 0.99302, 0.99305, 0.99243, 0.99247, 0.9916, 0.99164, 0.99085, 0.99087, 0.98985, 0.9899, 0.98857, 0.98859, 0.98717, 0.98721, 0.98551, 0.98556, 0.98301, 0.98305, 0.9805, 0.98055, 0.97635, 0.97641, 0.97183, 0.97187, 0.96496, 0.965, 0.95526, 0.95533, 0.94102, 0.94104, 0.91809, 0.91815, 0.87871, 0.87879, 0.80141, 0.80164]),
    "wan2.1_vace_14B": np.array([1.0]*2+[1.02504, 1.03017, 1.00025, 1.00251, 0.9985, 0.99962, 0.99779, 0.99771, 0.9966, 0.99658, 0.99482, 0.99476, 0.99467, 0.99451, 0.99664, 0.99656, 0.99434, 0.99431, 0.99533, 0.99545, 0.99468, 0.99465, 0.99438, 0.99434, 0.99516, 0.99517, 0.99384, 0.9938, 0.99404, 0.99401, 0.99517, 0.99516, 0.99409, 0.99408, 0.99428, 0.99426, 0.99347, 0.99343, 0.99418, 0.99416, 0.99271, 0.99269, 0.99313, 0.99311, 0.99215, 0.99215, 0.99218, 0.99215, 0.99216, 0.99217, 0.99163, 0.99161, 0.99138, 0.99135, 0.98982, 0.9898, 0.98996, 0.98995, 0.9887, 0.98866, 0.98772, 0.9877, 0.98767, 0.98765, 0.98573, 0.9857, 0.98501, 0.98498, 0.9838, 0.98376, 0.98177, 0.98173, 0.98037, 0.98035, 0.97678, 0.97677, 0.97546, 0.97543, 0.97184, 0.97183, 0.96711, 0.96708, 0.96349, 0.96345, 0.95629, 0.95625, 0.94926, 0.94929, 0.93964, 0.93961, 0.92511, 0.92504, 0.90693, 0.90678, 0.8796, 0.87945, 0.86111, 0.86189]),
    "wan2.2_t2v_14B": np.array([1.0]*2+[1.00124, 1.00155, 0.99822, 0.99851, 0.99696, 0.99687, 0.99703, 0.99732, 0.9966, 0.99679, 0.99602, 0.99658, 0.99578, 0.99664, 0.99484, 0.9949, 0.99633, 0.996, 0.99659, 0.99683, 0.99534, 0.99549, 0.99584, 0.99577, 0.99681, 0.99694, 0.99563, 0.99554, 0.9944, 0.99473, 0.99594, 0.9964, 0.99466, 0.99461, 0.99453, 0.99481, 0.99389, 0.99365, 0.99391, 0.99406, 0.99354, 0.99361, 0.99283, 0.99278, 0.99268, 0.99263, 0.99057, 0.99091, 0.99125, 0.99126, 0.65523, 0.65252, 0.98808, 0.98852, 0.98765, 0.98736, 0.9851, 0.98535, 0.98311, 0.98339, 0.9805, 0.9806, 0.97776, 0.97771, 0.97278, 0.97286, 0.96731, 0.96728, 0.95857, 0.95855, 0.94385, 0.94385, 0.92118, 0.921, 0.88108, 0.88076, 0.80263, 0.80181]),
    "wan2.2_ti2v_5B": np.array([1.0]*2+[0.99505, 0.99389, 0.99441, 0.9957, 0.99558, 0.99551, 0.99499, 0.9945, 0.99534, 0.99548, 0.99468, 0.9946, 0.99463, 0.99458, 0.9946, 0.99453, 0.99408, 0.99404, 0.9945, 0.99441, 0.99409, 0.99398, 0.99403, 0.99397, 0.99382, 0.99377, 0.99349, 0.99343, 0.99377, 0.99378, 0.9933, 0.99328, 0.99303, 0.99301, 0.99217, 0.99216, 0.992, 0.99201, 0.99201, 0.99202, 0.99133, 0.99132, 0.99112, 0.9911, 0.99155, 0.99155, 0.98958, 0.98957, 0.98959, 0.98958, 0.98838, 0.98835, 0.98826, 0.98825, 0.9883, 0.98828, 0.98711, 0.98709, 0.98562, 0.98561, 0.98511, 0.9851, 0.98414, 0.98412, 0.98284, 0.98282, 0.98104, 0.98101, 0.97981, 0.97979, 0.97849, 0.97849, 0.97557, 0.97554, 0.97398, 0.97395, 0.97171, 0.97166, 0.96917, 0.96913, 0.96511, 0.96507, 0.96263, 0.96257, 0.95839, 0.95835, 0.95483, 0.95475, 0.94942, 0.94936, 0.9468, 0.94678, 0.94583, 0.94594, 0.94843, 0.94872, 0.96949, 0.97015]),
    "wan2.2_i2v_14B": np.array([1.0]*2+[0.99191, 0.99144, 0.99356, 0.99337, 0.99326, 0.99285, 0.99251, 0.99264, 0.99393, 0.99366, 0.9943, 0.9943, 0.99276, 0.99288, 0.99389, 0.99393, 0.99274, 0.99289, 0.99316, 0.9931, 0.99379, 0.99377, 0.99268, 0.99271, 0.99222, 0.99227, 0.99175, 0.9916, 0.91076, 0.91046, 0.98931, 0.98933, 0.99087, 0.99088, 0.98852, 0.98855, 0.98895, 0.98896, 0.98806, 0.98808, 0.9871, 0.98711, 0.98613, 0.98618, 0.98434, 0.98435, 0.983, 0.98307, 0.98185, 0.98187, 0.98131, 0.98131, 0.9783, 0.97835, 0.97619, 0.9762, 0.97264, 0.9727, 0.97088, 0.97098, 0.96568, 0.9658, 0.96045, 0.96055, 0.95322, 0.95335, 0.94579, 0.94594, 0.93297, 0.93311, 0.91699, 0.9172, 0.89174, 0.89202, 0.8541, 0.85446, 0.79823, 0.79902]),
}


def magcache_flux_forward(
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
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        cur_step = transformer_options.get("current_step")
        
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

        # enable magcache
        # enable magcache
        if not hasattr(self, 'accumulated_err'):
            self.accumulated_err = 0
            self.accumulated_ratio = 1
            self.accumulated_steps = 0
        skip_forward = False
        if enable_magcache and cur_step not in [11]:
            cur_mag_ratio = mag_ratios[cur_step]
            self.accumulated_ratio = self.accumulated_ratio*cur_mag_ratio # magnitude ratio between current step and the cached step
            self.accumulated_steps += 1 # skip steps plus 1
            cur_skip_err = np.abs(1-self.accumulated_ratio) # skip error of current steps
            self.accumulated_err += cur_skip_err # accumulated error of multiple steps
            if self.accumulated_err<magcache_thresh and self.accumulated_steps<=magcache_K:
                skip_forward = True
                residual_x = self.residual_cache
            else:
                self.accumulated_err = 0
                self.accumulated_steps = 0
                self.accumulated_ratio = 1.0


        if skip_forward:
            img += self.residual_cache.to(img.device)
        else:
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
            self.residual_cache = (img - ori_img).to(mm.unet_offload_device())

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        
        return img

def magcache_ominigen2_forward(self, x, timesteps, context, num_tokens, ref_latents=None, attention_mask=None, **kwargs):
    B, C, H, W = x.shape
    hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    _, _, H_padded, W_padded = hidden_states.shape
    timestep = 1.0 - timesteps
    text_hidden_states = context
    text_attention_mask = attention_mask
    ref_image_hidden_states = ref_latents
    device = hidden_states.device

    temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

    (
        hidden_states, ref_image_hidden_states,
        img_mask, ref_img_mask,
        l_effective_ref_img_len, l_effective_img_len,
        ref_img_sizes, img_sizes,
    ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

    (
        context_rotary_emb, ref_img_rotary_emb, noise_rotary_emb,
        rotary_emb, encoder_seq_lengths, seq_lengths,
    ) = self.rope_embedder(
        hidden_states.shape[0], text_hidden_states.shape[1], [num_tokens] * text_hidden_states.shape[0],
        l_effective_ref_img_len, l_effective_img_len,
        ref_img_sizes, img_sizes, device,
    )

    for layer in self.context_refiner:
        text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)

    img_len = hidden_states.shape[1]
    combined_img_hidden_states = self.img_patch_embed_and_refine(
        hidden_states, ref_image_hidden_states,
        img_mask, ref_img_mask,
        noise_rotary_emb, ref_img_rotary_emb,
        l_effective_ref_img_len, l_effective_img_len,
        temb,
    )

    hidden_states = torch.cat([text_hidden_states, combined_img_hidden_states], dim=1)
    attention_mask = None

    for layer in self.layers:
        hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)

    hidden_states = self.norm_out(hidden_states, temb)

    p = self.patch_size
    output = rearrange(hidden_states[:, -img_len:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',  h=H_padded // p, w=W_padded// p, p1=p, p2=p)[:, :, :H, :W]

    return -output

def magcache_hunyuanvideo_forward(
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
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        cur_step = transformer_options.get("current_step")
        
        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        if ref_latent is not None:
            ref_latent_ids = self.img_ids(ref_latent)
            ref_latent = self.img_in(ref_latent)
            img = torch.cat([ref_latent, img], dim=-2)
            ref_latent_ids[..., 0] = -1
            ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
            img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)

        if guiding_frame_index is not None:
            token_replace_vec = self.time_in(timestep_embedding(guiding_frame_index, 256, time_factor=1.0))
            vec_ = self.vector_in(y[:, :self.params.vec_in_dim])
            vec = torch.cat([(vec_ + token_replace_vec).unsqueeze(1), (vec_ + vec).unsqueeze(1)], dim=1)
            frame_tokens = (initial_shape[-1] // self.patch_size[-1]) * (initial_shape[-2] // self.patch_size[-2])
            modulation_dims = [(0, frame_tokens, 0), (frame_tokens, None, 1)]
            modulation_dims_txt = [(0, None, 1)]
        else:
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
            modulation_dims = None
            modulation_dims_txt = None

        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

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

        # enable magcache
        if not hasattr(self, 'accumulated_err'):
            self.accumulated_err = 0
            self.accumulated_ratio = 1
            self.accumulated_steps = 0
        skip_forward = False
        if enable_magcache:
            cur_mag_ratio = mag_ratios[cur_step]
            self.accumulated_ratio = self.accumulated_ratio*cur_mag_ratio # magnitude ratio between current step and the cached step
            self.accumulated_steps += 1 # skip steps plus 1
            cur_skip_err = np.abs(1-self.accumulated_ratio) # skip error of current steps
            self.accumulated_err += cur_skip_err # accumulated error of multiple steps
            if self.accumulated_err<magcache_thresh and self.accumulated_steps<=magcache_K:
                skip_forward = True
                residual_x = self.residual_cache
            else:
                self.accumulated_err = 0
                self.accumulated_steps = 0
                self.accumulated_ratio = 1.0

        if skip_forward:
            img += self.residual_cache.to(img.device)
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

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
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:, : img_len]
            self.residual_cache = (img - ori_img).to(mm.unet_offload_device())

        if ref_latent is not None:
            img = img[:, ref_latent.shape[1]:]
        
        img = self.final_layer(img, vec, modulation_dims=modulation_dims)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
        return img

def magcache_hunyuanvideo15_forward(
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
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        total_infer_steps = transformer_options.get("total_infer_steps")
        cur_step = transformer_options.get("current_step")
        
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
        
        # MagCache initialization
        if not hasattr(self, 'accumulated_err'):
            # forward conditional and unconditional seperately.
            self.accumulated_err = [0.0, 0.0]
            self.accumulated_ratio = [1.0, 1.0]
            self.accumulated_steps = [0, 0]
            self.residual_cache = [None, None]
            self.cnt = 0            
        skip_forward = False
        if enable_magcache:  # Skip certain steps if needed
            cur_mag_ratio = mag_ratios[2*cur_step+self.cnt%2]
            self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2] * cur_mag_ratio
            self.accumulated_steps[self.cnt%2] += 1
            cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt%2])
            self.accumulated_err[self.cnt%2] += cur_skip_err
            if self.accumulated_err[self.cnt%2] < magcache_thresh and self.accumulated_steps[self.cnt%2] <= magcache_K:
                skip_forward = True
            else:
                self.accumulated_err[self.cnt%2] = 0
                self.accumulated_steps[self.cnt%2] = 0
                self.accumulated_ratio[self.cnt%2] = 1.0
                
        if skip_forward:
            # print("skip step: ", self.cnt, self.accumulated_err[self.cnt%2], self.accumulated_steps[self.cnt%2])
            img += self.residual_cache[self.cnt%2].to(img.device)
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims_img': modulation_dims, 'modulation_dims_txt': modulation_dims_txt}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims_img=modulation_dims, modulation_dims_txt=modulation_dims_txt)

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
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask, 'modulation_dims': modulation_dims}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask, modulation_dims=modulation_dims)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:, : img_len]
            self.residual_cache[self.cnt%2] = (img - ori_img).to(mm.unet_offload_device())
        self.cnt += 1
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


def magcache_wanmodel_forward(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        # ret_ratio = transformer_options.get("magcache_ret_ratio")
        enable_magcache = transformer_options.get("enable_magcache", True)
        cur_step = transformer_options.get("current_step")
        mag_ratios = transformer_options.get("mag_ratios")
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        blocks_replace = patches_replace.get("dit", {})

        # enable magcache
        if not hasattr(self, 'magcache_state'):
            self.magcache_state = {
                0: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None}, # condition
                1: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None} # uncondition
            }

        def update_cache_state(cache, cur_step):
            if enable_magcache:
                cur_scale = mag_ratios[cur_step]
                cache['accumulated_ratio'] = cache['accumulated_ratio']*cur_scale
                cache['accumulated_steps'] = cache['accumulated_steps'] + 1
                cache['accumulated_err'] += np.abs(1-cache['accumulated_ratio'])
                if cache['accumulated_err']<=magcache_thresh and cache['accumulated_steps']<=magcache_K:
                    cache['skip_forward'] = True
                else:
                    cache['skip_forward'] = False
                    cache['accumulated_ratio'] = 1.0
                    cache['accumulated_steps'] = 0
                    cache['accumulated_err'] = 0
            
        b = int(len(x) / len(cond_or_uncond))

        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.magcache_state[k], cur_step*2+i)

        if enable_magcache:
            skip_forward = False
            for k in cond_or_uncond: #Skip or keep the uncondtional and conditional forward together, which may be different from the original official implementation in MagCache.
                skip_forward = (skip_forward or self.magcache_state[k]['skip_forward'])
        else:
            skip_forward = False

        if skip_forward:
            for i, k in enumerate(cond_or_uncond):
                if self.magcache_state[k]['residual_cache'] is not None:
                    x[i*b:(i+1)*b] += self.magcache_state[k]['residual_cache'].to(x.device)
        else:
            ori_x = x.clone()
            for i, block in enumerate(self.blocks): # note: perform conditional and uncondition forward together, which can be improved by seperate into two single process.
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap, "transformer_options": transformer_options})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
            for i, k in enumerate(cond_or_uncond):
                self.magcache_state[k]['residual_cache'] = (x - ori_x)[i*b:(i+1)*b].to(mm.unet_offload_device())

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

def magcache_wan_vace_forward(
        self,
        x,
        t,
        context,
        vace_context,
        vace_strength,
        clip_fea=None,
        freqs=None,
        transformer_options={},
        **kwargs,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        cond_or_uncond = transformer_options.get("cond_or_uncond")
        enable_magcache = transformer_options.get("enable_magcache", True)
        cur_step = transformer_options.get("current_step")
        mag_ratios = transformer_options.get("mag_ratios")
        # embeddings
        x = self.patch_embedding(x.float()).to(x.dtype)
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # context
        context = self.text_embedding(context)

        context_img_len = None
        if clip_fea is not None:
            if self.img_emb is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)
            context_img_len = clip_fea.shape[-2]

        orig_shape = list(vace_context.shape)
        vace_context = vace_context.movedim(0, 1).reshape([-1] + orig_shape[2:])
        c = self.vace_patch_embedding(vace_context.float()).to(vace_context.dtype)
        c = c.flatten(2).transpose(1, 2)
        c = list(c.split(orig_shape[0], dim=0))

        # arguments
        x_orig = x

        # enable magcache
        if not hasattr(self, 'magcache_state'):
            self.magcache_state = {
                0: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None}, # condition
                1: {'skip_forward': False, 'accumulated_ratio': 1, 'accumulated_err': 0, 'accumulated_steps': 0, 'residual_cache': None} # uncondition
            }
        b = int(len(x) / len(cond_or_uncond))
        def update_cache_state(cache, cur_step):
            if enable_magcache:
                cur_scale = mag_ratios[cur_step]
                cache['accumulated_ratio'] = cache['accumulated_ratio']*cur_scale
                cache['accumulated_steps'] = cache['accumulated_steps'] + 1
                cache['accumulated_err'] += np.abs(1-cache['accumulated_ratio'])
                if cache['accumulated_err']<=magcache_thresh and cache['accumulated_steps']<=magcache_K:
                    cache['skip_forward'] = True
                else:
                    cache['skip_forward'] = False
                    cache['accumulated_ratio'] = 1.0
                    cache['accumulated_steps'] = 0
                    cache['accumulated_err'] = 0
        
        for i, k in enumerate(cond_or_uncond):
            update_cache_state(self.magcache_state[k], cur_step*2+i)
        skip_forward = False
        if enable_magcache:
            for k in cond_or_uncond: #Skip or keep the uncondtional and conditional forward together, which may be different from the original official implementation in MagCache.
                skip_forward = (skip_forward or self.magcache_state[k]['skip_forward'])

        if skip_forward:
            for i, k in enumerate(cond_or_uncond):
                if self.magcache_state[k]['residual_cache'] is not None:
                    x[i*b:(i+1)*b] += self.magcache_state[k]['residual_cache'].to(x.device)
        else:
            for i, block in enumerate(self.blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], context=args["txt"], e=args["vec"], freqs=args["pe"], context_img_len=context_img_len)
                        return out
                    out = blocks_replace[("double_block", i)]({"img": x, "txt": context, "vec": e0, "pe": freqs}, {"original_block": block_wrap})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)

                ii = self.vace_layers_mapping.get(i, None)
                if ii is not None:
                    for iii in range(len(c)):
                        c_skip, c[iii] = self.vace_blocks[ii](c[iii], x=x_orig, e=e0, freqs=freqs, context=context, context_img_len=context_img_len)
                        x += c_skip * vace_strength[iii]
                    del c_skip
            for i, k in enumerate(cond_or_uncond):
                self.magcache_state[k]['residual_cache'] = (x - x_orig)[i*b:(i+1)*b].to(mm.unet_offload_device())
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x

def magcache_qwen_image_forward(
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
        
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        total_infer_steps = transformer_options.get("total_infer_steps")

        # MagCache initialization
        if not hasattr(self, 'accumulated_err'):
            # forward conditional and unconditional seperately.
            self.accumulated_err = [0.0, 0.0]
            self.accumulated_ratio = [1.0, 1.0]
            self.accumulated_steps = [0, 0]
            self.residual_cache = [None, None]
            self.cnt = 0            
        skip_forward = False
        if enable_magcache:  # Skip certain steps if needed
            cur_mag_ratio = mag_ratios[self.cnt]
            self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2] * cur_mag_ratio
            self.accumulated_steps[self.cnt%2] += 1
            cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt%2])
            self.accumulated_err[self.cnt%2] += cur_skip_err
            if self.accumulated_err[self.cnt%2] < magcache_thresh and self.accumulated_steps[self.cnt%2] <= magcache_K:
                skip_forward = True
            else:
                self.accumulated_err[self.cnt%2] = 0
                self.accumulated_steps[self.cnt%2] = 0
                self.accumulated_ratio[self.cnt%2] = 1.0
                
        if skip_forward:
            hidden_states += self.residual_cache[self.cnt%2]
        else:
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
            self.residual_cache[self.cnt%2] = hidden_states - ori_hidden_states
        self.cnt += 1
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]

def magcache_chroma_forward(
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
        magcache_thresh = transformer_options.get("magcache_thresh")
        magcache_K = transformer_options.get("magcache_K")
        mag_ratios = transformer_options.get("mag_ratios")
        enable_magcache = transformer_options.get("enable_magcache", False)
        total_infer_steps = transformer_options.get("total_infer_steps")

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
        
        # MagCache initialization
        if not hasattr(self, 'accumulated_err'):
            # forward conditional and unconditional seperately.
            self.accumulated_err = [0.0, 0.0]
            self.accumulated_ratio = [1.0, 1.0]
            self.accumulated_steps = [0, 0]
            self.residual_cache = [None, None]
            self.cnt = 0            
        skip_forward = False
        if enable_magcache:  # Skip certain steps if needed
            cur_mag_ratio = mag_ratios[self.cnt]
            self.accumulated_ratio[self.cnt%2] = self.accumulated_ratio[self.cnt%2] * cur_mag_ratio
            self.accumulated_steps[self.cnt%2] += 1
            cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt%2])
            self.accumulated_err[self.cnt%2] += cur_skip_err
            if self.accumulated_err[self.cnt%2] < magcache_thresh and self.accumulated_steps[self.cnt%2] <= magcache_K:
                skip_forward = True
            else:
                self.accumulated_err[self.cnt%2] = 0
                self.accumulated_steps[self.cnt%2] = 0
                self.accumulated_ratio[self.cnt%2] = 1.0
                
        if skip_forward:
            
            img += self.residual_cache[self.cnt%2].to(img.device)
        else:
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
            self.residual_cache[self.cnt%2] = (img - ori_img).to(mm.unet_offload_device())
        self.cnt += 1
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)
        return img

class MagCache:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the MagCache will be applied to."}),
                "model_type": (["flux", "flux_kontext", "chroma", "qwen_image", "hunyuan_video", "hunyuan_video1.5", "wan2.1_t2v_1.3B", "wan2.1_t2v_14B", "wan2.1_i2v_480p_14B", "wan2.1_i2v_720p_14B", "wan2.1_vace_1.3B", "wan2.1_vace_14B"], {"default": "wan2.1_t2v_1.3B", "tooltip": "Supported diffusion model."}),
                "magcache_thresh": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.3, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.1, "max": 0.3, "step": 0.01, "tooltip": "The start percentage of the steps that will apply MagCache."}),
                "magcache_K": ("INT", {"default": 2, "min": 0, "max": 6, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
                "end_step": ("INT", {"default": -1, "min": -100, "max": 100, "step": 1, "tooltip": "The maxium skip steps of MagCache."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_magcache"
    CATEGORY = "MagCache"
    TITLE = "MagCache"
    
    def apply_magcache(self, model, model_type: str, magcache_thresh: float, retention_ratio: float, magcache_K: int, start_step: int, end_step:int):
        if magcache_thresh == 0:
            return (model,)

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        new_model.model_options["transformer_options"]["magcache_thresh"] = magcache_thresh
        new_model.model_options["transformer_options"]["retention_ratio"] = retention_ratio
        mag_ratios = SUPPORTED_MODELS_MAG_RATIOS[model_type]
        mag_ratios_tensor = torch.from_numpy(mag_ratios).float()
        new_model.model_options["transformer_options"]["mag_ratios"] = mag_ratios_tensor
        new_model.model_options["transformer_options"]["magcache_K"] = magcache_K
        new_model.model_options["transformer_options"]["start_step"] = start_step
        new_model.model_options["transformer_options"]["end_step"] = end_step
        diffusion_model = new_model.get_model_object("diffusion_model")

        if "flux" in model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_flux_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "chroma" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_chroma_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "qwen_image" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                _forward=magcache_qwen_image_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "hunyuan_video" == model_type:
            is_cfg = False
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_hunyuanvideo_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "hunyuan_video1.5" == model_type:
            is_cfg = True # only support cfg>1
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_hunyuanvideo15_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "wan2.1_vace" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_wan_vace_forward.__get__(diffusion_model, diffusion_model.__class__)
            ) 
        elif "wan2.1" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
            )
        elif "wan2.2" in model_type:
            is_cfg = True
            context = patch.multiple(
                diffusion_model,
                forward_orig=magcache_wanmodel_forward.__get__(diffusion_model, diffusion_model.__class__)
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
                if is_cfg and "wan" in model_type:
                    # uncond first
                    if (1 in cond_or_uncond) and hasattr(diffusion_model, 'magcache_state'):
                        delattr(diffusion_model, 'magcache_state')
                else:
                    if hasattr(diffusion_model, 'accumulated_err'):
                        delattr(diffusion_model, 'accumulated_err')
            
            total_infer_steps = len(sigmas)-1
            
            if model_type == "hunyuan_video1.5" and total_infer_steps>20:
                mag_ratios = SUPPORTED_MODELS_MAG_RATIOS["hunyuan_video1.5_40steps"]
                mag_ratios_tensor = torch.from_numpy(mag_ratios).float()
                c["transformer_options"]["mag_ratios"] = mag_ratios_tensor
            
            start_step = c["transformer_options"]["start_step"]
            end_step = c["transformer_options"]["end_step"]
            if end_step<0:
                end_step = total_infer_steps + end_step
            if  current_step_index>=int(total_infer_steps*c["transformer_options"]["retention_ratio"]) and (start_step<=current_step_index<=end_step): # start index of magcache
                c["transformer_options"]["enable_magcache"] = True
            else:
                c["transformer_options"]["enable_magcache"] = False
            calibration_len = len(c["transformer_options"]["mag_ratios"])//2 if is_cfg else len(c["transformer_options"]["mag_ratios"])
            c["transformer_options"]["current_step"] = current_step_index if (total_infer_steps)==calibration_len else int((current_step_index*((calibration_len-1)/(total_infer_steps-1)))) #interpolate when the steps is not equal to pre-defined steps
            # if "chroma" in model_type:
            #     predefined_steps = len(c["transformer_options"]["mag_ratios"])//2
            #     assert total_infer_steps==predefined_steps, f"The inference steps of chroma must be {predefined_steps}."
            
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

class CompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the torch.compile will be applied to."}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "backend": (["inductor","cudagraphs", "eager", "aot_eager"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_compile"
    CATEGORY = "MagCache"
    TITLE = "Compile Model"
    
    def apply_compile(self, model, mode: str, backend: str, fullgraph: bool, dynamic: bool):
        patch_optimized_module()
        patch_same_meta()
        torch._dynamo.config.suppress_errors = True
        
        new_model = model.clone()
        new_model.add_object_patch(
                                "diffusion_model",
                                torch.compile(
                                    new_model.get_model_object("diffusion_model"),
                                    mode=mode,
                                    backend=backend,
                                    fullgraph=fullgraph,
                                    dynamic=dynamic
                                )
                            )
        
        return (new_model,)
    
    
NODE_CLASS_MAPPINGS = {
    "MagCache": MagCache,
    "CompileModel": CompileModel
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}

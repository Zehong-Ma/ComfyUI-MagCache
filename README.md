# ComfyUI-MagCache

## 🫖 Introduction 
Magnitude-aware Cache (MagCache) is a training-free caching approach. It estimates the fluctuating differences among model outputs across timesteps based on the robust **magnitude observations**, and thereby accelerating the inference using the error modeling mechanism and adaptive cache strategy. MagCache works well for Video Diffusion Models, Image Diffusion models. For more details and results, please visit our [project page](https://zehong-ma.github.io/MagCache) and [code](https://github.com/Zehong-Ma/MagCache).

MagCache has now been integrated into ComfyUI and is compatible with the ComfyUI native nodes. ComfyUI-MagCache is easy to use, simply connect the MagCache node with the ComfyUI native nodes for seamless usage.

## 🔥 Latest News 
- **If you like our project, please give us a star ⭐ on GitHub for the latest update.**
- [2025/11/23] 🔥 Support [Qwen-Image](https://github.com/QwenLM/Qwen-Image) officially, achieving a 1.75x acceleration.
- [2025/11/22] 🔥 Support [HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) officially, achieving a 1.7x acceleration.
- [2025/7/2] 🔥 Support [Wan2.1-VACE-14B](https://github.com/ali-vilab/VACE) officially. Thanks @[Qentah](https://github.com/Qentah).
- [2025/6/30] 🔥 Support [Flux-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) with 2x speedup. Please see the demo [here](https://github.com/user-attachments/assets/79d5f654-5828-442d-b1a1-9b754c17e457).
- [2025/6/19] 🔥 Support [FramePack](https://github.com/lllyasviel/FramePack) with Gradio Demo in [MagCache-FramePack](https://github.com/Zehong-Ma/MagCache).
- [2025/6/18] 👉 We're collecting the best parameter settings from the community. <br>     👉**Open this [discussion issue](https://github.com/Zehong-Ma/ComfyUI-MagCache/issues/15) to contribute your configuration!**
- [2025/6/17] 🔥 Support [Wan2.1-VACE-1.3B](https://github.com/ali-vilab/VACE), achieving a 1.7× acceleration. 
- [2025/6/17] 🔥 MagCache is supported by [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). Thanks @[kijai](https://github.com/kijai). 
- [2025/6/16] 🔥 Support [Chroma](https://huggingface.co/lodestones/Chroma). Thanks @[kabachuha](https://github.com/kabachuha) for the codebase.
- [2025/6/10] 🔥 Support [Wan2.1](https://github.com/Wan-Video/Wan2.1) T2V&I2V, [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) T2V, [FLUX-dev]((https://github.com/black-forest-labs/flux)) T2I

## Installation
<!-- Installation via ComfyUI-Manager is preferred. Simply search for ComfyUI-MagCache in the list of nodes and click install.
### Manual installation -->
1. Go to comfyUI custom_nodes folder, `ComfyUI/custom_nodes/`
2. git clone https://github.com/zehong-ma/ComfyUI-MagCache.git
3. Go to ComfyUI-MagCache folder, `cd ComfyUI-MagCache/`
4. pip install -r requirements.txt
5. Go to the project folder `ComfyUI/` and run `python main.py`
## Usage

### Download Model Weights
Please first to prepare the model weights in ComfyUI format by referring to the follow links:
- [Wan2.1](https://comfyanonymous.github.io/ComfyUI_examples/wan/)
- [HunyuanVideo](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)
- [FLUX](https://comfyanonymous.github.io/ComfyUI_examples/flux/)
- [Chroma](https://huggingface.co/lodestones/Chroma)

### MagCache

**We're collecting the best parameter settings from the community. Open this [discussion issue](https://github.com/Zehong-Ma/ComfyUI-MagCache/issues/15) to contribute your configuration!**

To use MagCache node, simply add `MagCache` node to your workflow after `Load Diffusion Model` node or `Load LoRA` node (if you need LoRA). Generally, MagCache can achieve a speedup of 2x to 3x with acceptable visual quality loss. The following table gives the recommended magcache_thresh, retention_ratio and magcache_K ​for different models:

<div align="center">

| Models                       |   magcache_thresh |   retention_ratio |    magcache_K     |  
|:----------------------------:|:-----------------:|:-----------------:|:-----------------:|
| FLUX                         |        0.24       |         0.1       |         5         |
| FLUX-Kontext                 |        0.05       |         0.2       |         4         |
| Chroma                       |        0.10       |         0.25      |         2         |
| Qwen-Image                   |        0.10       |         0.20      |         2         |
| HunyuanVideo-T2V             |        0.24       |         0.2       |         6         |
| HunyuanVideo1.5-T2V          |        0.03       |         0.25      |         2         |
| Wan2.1-T2V-1.3B              |        0.12       |         0.2       |         4         |
| Wan2.1-T2V-14B               |        0.24       |         0.2       |         6         |
| Wan2.1-I2V-480P-14B          |        0.24       |         0.2       |         6         |
| Wan2.1-I2V-720P-14B          |        0.24       |         0.2       |         6         |
| Wan2.1-VACE-1.3B             |        0.02       |         0.2       |         3         |
| Wan2.1-VACE-14B              |        0.02       |         0.2       |         3         |

</div>

**If the image/video after applying MagCache is of low quality, please decrease `magcache_thresh` and `magcache_K`**. The default parameters are configured for extremely fast inference(2x-3x), which may lead to failures in some cases.

The demo workflows ([flux](./examples/flux.json), [flux-kontext](./examples/flux_1_kontext_dev.json), [qwen-image](./examples/qwen_image_basic_example.json), [chroma](./examples/chroma.json), [hunyuanvideo](./examples/hunyuanvideo.json), [hunyuanvideo1.5](./examples/video_hunyuan_video_1.5_720p_t2v_magcache_fast.json), [wan2.1_t2v](./examples/wan2.1_t2v.json), [wan2.1_i2v](./examples/wan2.1_i2v.json), and [wan2.1_vace](./examples/wan2.1_vace.json)) are placed in examples folder. The workflow [chroma_calibration](./examples/chroma_calibration.json) is used to calibrate the `mag_ratios` for `Chroma` when the number of inference steps differs from the pre-defined value.
**In our experiments, the videos generated by Wan2.1 are not as high-quality as those produced by the [original unquantized version](https://github.com/Wan-Video/Wan2.1).**


### Compile Model
To use Compile Model node, simply add `Compile Model` node to your workflow after `Load Diffusion Model` node or `MagCache` node. Compile Model uses `torch.compile` to enhance the model performance by compiling model into more efficient intermediate representations (IRs). This compilation process leverages backend compilers to generate optimized code, which can significantly speed up inference. The compilation may take long time when you run the workflow at first, but once it is compiled, inference is extremely fast. 
<!-- The usage is shown below: -->
<!-- ![](./assets/compile.png) -->

## Acknowledgments
Thanks to [ComfyUI-TeaCache](https://github.com/welltop-cn/ComfyUI-TeaCache), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [ComfyUI-MagCache](https://github.com/wildminder/ComfyUI-MagCache), [MagCache](https://github.com/Zehong-Ma/MagCache/), [TeaCache](https://github.com/ali-vilab/TeaCache), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [FLUX](https://github.com/black-forest-labs/flux), [Chroma](https://huggingface.co/lodestones/Chroma), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), and [Wan2.1](https://github.com/Wan-Video/Wan2.1).

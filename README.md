# DyMO: Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling
[Xin Xie](https://shelsin.github.io/), [Dong Gong](https://donggong1.github.io/)

<a href="https://arxiv.org/abs/2412.00759"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" height=22.5></a>
<a href="https://shelsin.github.io/dymo.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" height=22.5></a>
<a href="https://www.youtube.com/watch?v=nKPAmAzJWFU"><img src="https://img.shields.io/badge/YouTube-Video-yellow?style=for-the-badge" height=22.5></a>

This is the official implementation of DyMO, introduced in [Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling](https://arxiv.org/abs/2412.00759).

## TODO
- [x] Release the inference code of DyMO (Coming soon/Almost there🤓).

## Abstract
<p>
Text-to-image diffusion model alignment is critical for improving the alignment between the generated images and human preferences. While training-based methods are constrained by high computational costs and dataset requirements, training-free alignment methods remain underexplored and are often limited by inaccurate guidance.
</p>
<p>
We propose a plug-and-play training-free alignment method, DyMO, for aligning the generated images and human preferences during inference. Apart from text-aware human preference scores, we introduce a semantic alignment objective for enhancing the semantic alignment in the early stages of diffusion, relying on the fact that the attention maps are effective reflections of the semantics in noisy images. We propose dynamic scheduling of multiple objectives and intermediate recurrent steps to reflect the requirements at different steps.
</p>
<p>
Experiments with diverse pre-trained diffusion models and metrics demonstrate the effectiveness and robustness of the proposed method. The project page: https://shelsin.github.io/dymo.github.io/
</p>

## Method Overview
![method_overview](assets/method.png)

## :unlock: Available Checkpoints
[Step-Aware_Preference_Models](https://huggingface.co/SPO-Diffusion-Models/Step-Aware_Preference_Models)

Download the model "sd-v1-5_step-aware_preference_model.bin" and "sdxl_step-aware_preference_model.bin" first, and then duplicate them to the "model_ckpts" file.

## :wrench: Inference

SD v1.5 inference
```bash
python inference_sd15_dymo.py
```

SDXL inference
```bash
python inference_sdxl_dymo.py
```

## :rocket: Acknowledgement
Our codebase references the code from [Diffusers](https://github.com/huggingface/diffusers), [SPO](https://rockeycoss.github.io/spo.github.io/) and [PickScore](https://github.com/yuvalkirstain/PickScore). We extend our gratitude to their authors for open-sourcing their code.

## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```
@InProceedings{xin2025dymo,
    author={Xie, Xin and Gong, Dong},
    title={DyMO: Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2025}
}
```

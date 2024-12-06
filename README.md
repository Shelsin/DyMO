# DyMO: Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling
[Xin Xie](https://shelsin.github.io/), [Dong Gong](https://donggong1.github.io/)

<a href="https://arxiv.org/abs/2412.00759"><img src="https://img.shields.io/badge/Paper-arXiv-red?style=for-the-badge" height=22.5></a>
<a href="https://shelsin.github.io/dymo.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" height=22.5></a>
<a href="https://www.youtube.com/watch?v=nKPAmAzJWFU"><img src="https://img.shields.io/badge/YouTube-Video-yellow?style=for-the-badge" height=22.5></a>

This is the official implementation of DyMO, introduced in [Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling](https://arxiv.org/abs/2412.00759).

## TODO
Release the inference code of DyMO (Coming soon).

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


## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```
@article{xin2024dymo,
  title={DyMO: Training-Free Diffusion Model Alignment with Dynamic Multi-Objective Scheduling},
  author={Xie, Xin and Gong, Dong},
  journal={arXiv preprint arXiv:2412.00759},
  year={2024}
}
```

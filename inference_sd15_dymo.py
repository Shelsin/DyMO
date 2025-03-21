import argparse
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
import tqdm
from diffusers import DDIMScheduler
import numpy as np
from PIL import Image
from einops import rearrange
from ddim_with_logprob import ddim_step_with_logprob_dymo, ddim_step_forward_from_xt_1_to_xt
from torch import nn
from diffusers.utils.torch_utils import randn_tensor
import math
import os
import random
from torchvision import transforms
from preference_models import get_preference_model_func, get_compare_func
from typing import Any, Callable, Dict, Optional, Union, List, Tuple
import spacy
from torch.nn import functional as F
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_attend_and_excite.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor,
)
import matplotlib.pyplot as plt
from DiffAugment import DiffAugment
import json

from utils import *
from format_pos_neg_pairs import process_p_n_groups

POLICY = 'color,translation,resize,cutout'


###########

def augment_image(image):
    augmentations = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.RandomRotation(degrees=15),
    ]
    return [aug(image) for aug in augmentations]


def register_attention_control(pipeline):
    attn_res = (16, 16)
    pipeline.attention_store = AttentionStore(attn_res)
    attn_procs = {}
    cross_att_count = 0
    for name in pipeline.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = AttendExciteAttnProcessor(
            attnstore=pipeline.attention_store, place_in_unet=place_in_unet
        )

    pipeline.unet.set_attn_processor(attn_procs)
    pipeline.attention_store.num_att_layers = cross_att_count
    return pipeline


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized

def get_attention_maps_list(attention_maps: torch.Tensor) -> List[torch.Tensor]:
    attention_maps *= 100
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]

    return attention_maps_list


def optimize_step_dymo(args, latents, pipeline, t, pred_x0_prev, prompt_embeds,
                      do_classifier_free_guidance, cross_attention_kwargs, preference_model_fn, extra_info,
                      generator, contrast_pairs, extra_step_kwargs):
    latents_grad = latents.detach().requires_grad_(True)
    with torch.enable_grad():
        latent_model_input = torch.cat([latents_grad] * 2) if do_classifier_free_guidance else latents_grad
        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = pipeline.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        attention_maps = pipeline.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_maps_list = get_attention_maps_list(attention_maps=attention_maps)

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred_text - noise_pred_uncond)
            correction = noise_pred_text - noise_pred_uncond

        if do_classifier_free_guidance and args.cfg_rescale > 0.0:
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                           guidance_rescale=args.cfg_rescale)

        latents_prev_sample, pred_x0_latents, log_prob = ddim_step_with_logprob_dymo(pipeline.scheduler, noise_pred,
                                                                                    t, latents_grad,
                                                                                    variance_noise=None,
                                                                                    **extra_step_kwargs)

        pred_x0_dno = pipeline.vae.decode(
            pred_x0_latents.to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]

        pred_x0_dno_no_grad = pred_x0_dno.clone().detach()


        replicate = 10
        pred_x0_dno_images = DiffAugment(pred_x0_dno.repeat(replicate, 1, 1, 1), policy=POLICY)
        pred_x0_dno_images = torch.cat([pred_x0_dno, pred_x0_dno_images], 0)
        extra_info['timesteps'] = t.repeat(pred_x0_dno_images.shape[0])
        preference_model_input_ids = [extra_info['input_ids'] for _ in range(pred_x0_dno_images.shape[0])]
        preference_model_input_ids = torch.cat(preference_model_input_ids, dim=0)
        extra_info['input_ids'] = preference_model_input_ids
        loss_preference = preference_model_fn(pred_x0_dno_images, extra_info)
        loss_preference = -torch.linalg.norm(loss_preference)

        # positive-negative loss
        positive_groups, negative_pairs = contrast_pairs["pos"], contrast_pairs["neg"]
        loss_pos = torch.tensor(0.0).cuda()
        for (i, j) in positive_groups:
            if i >= 77:
                continue
            elif j >= 77:
                continue
            loss_pos += cos_dist(attention_maps_list[i], attention_maps_list[j])

        loss_neg = torch.tensor(0.0).cuda()
        for (i, j) in negative_pairs:
            if i >= 77:
                continue
            elif j >= 77:
                continue
            loss_neg -= cos_dist(attention_maps_list[i], attention_maps_list[j])

        loss_cos = loss_pos / (len(positive_groups) + 1e-6) + loss_neg / (len(negative_pairs) + 1e-6)

        if pred_x0_prev is not None:
            delta_content_relative = torch.norm(normal(pred_x0_latents) - normal(pred_x0_prev), p=2) / torch.norm(normal(pred_x0_prev), p=2)
            k = 1
            alpha_attn = 10 * (1 - torch.exp(-k * delta_content_relative))
            alpha_preference = torch.exp(-k * delta_content_relative)

        if args.flag_dpo_loss:
            total_loss = loss_cos * 10

        elif args.flag_attn_loss:
            total_loss = loss_cos * alpha_attn + loss_preference * alpha_preference
        else:
            total_loss = loss_preference

        grad_direction = torch.autograd.grad(outputs=total_loss, inputs=latents_grad)[0]
        if args.flag_dpo_loss:
            if (grad_direction * grad_direction).sum().sqrt().item() == 0.0:
                rho = 0.01
            else:
                rho = (correction * correction).sum().sqrt().item() * args.cfg_scale / (grad_direction * grad_direction).sum().sqrt().item() * 0.2
        elif args.flag_attn_loss:
            rho = (correction * correction).sum().sqrt().item() / (grad_direction * grad_direction).sum().sqrt().item()
        else:
            rho = (correction * correction).sum().sqrt().item() / (grad_direction * grad_direction).sum().sqrt().item()

        latents_prev_sample = latents_prev_sample - args.opt_stength * rho * grad_direction.detach()

        if args.flag_round_stop:
            grad_abs = torch.norm(grad_direction.detach(), p=2) * args.num_round
            args.num_optimization = int(grad_abs)
            args.flag_round_stop = False
            # print("num_optimization", args.num_optimization)

        return latents_prev_sample.detach(), pred_x0_latents.detach(), log_prob.detach(), grad_direction.detach(), pred_x0_dno_no_grad


def main(args, prompt, save_name, pipeline, preference_model_fn, contrast_pairs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_dtype = torch.float16

    # random seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)


    pipeline.to(device)
    pipeline.unet = pipeline.unet.eval()
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    do_classifier_free_guidance = args.cfg_scale > 1.0

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
    )[0]

    prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(device)
    )[0]

    prompt_input_ids = pipeline.tokenizer(
        prompt,
        max_length=pipeline.tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    # the size of generated image
    height = args.height
    width = args.width

    pipeline.scheduler.set_timesteps(args.num_timesteps, device=device)
    timesteps = pipeline.scheduler.timesteps

    # Prepare latent variables
    num_channels_latents = pipeline.unet.config.in_channels

    height = height or pipeline.unet.config.sample_size * pipeline.vae_scale_factor
    width = width or pipeline.unet.config.sample_size * pipeline.vae_scale_factor

    cross_attention_kwargs = {}
    text_encoder_lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None

    prompt_embeds = pipeline._encode_prompt(
        None,
        device,
        1,
        do_classifier_free_guidance,
        None,
        prompt_embeds=prompt_embed,
        negative_prompt_embeds=neg_prompt_embed,
        lora_scale=text_encoder_lora_scale,
    )

    latents_ori = pipeline.prepare_latents(
        1,  # num_images_per_prompt
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None
    )

    latents = latents_ori.clone().detach()

    eta = 1.0
    extra_step_kwargs = pipeline.prepare_extra_step_kwargs(generator, eta)
    extra_info = {}

    pred_x0_prev = None

    with pipeline.progress_bar(total=timesteps.shape[0]) as progress_bar:
        for i, t in enumerate(timesteps):
            latents_prev_sample = None

            args.num_optimization = 1
            args.flag_optimize = True
            args.flag_round = True
            if i >= args.time2:
                args.flag_attn_loss = False
                args.flag_dpo_loss = False
                args.num_round = 3
            elif (i >= 0) and (i < args.time1):
                args.flag_dpo_loss = True
                args.flag_attn_loss = False
                args.num_round = 1
                if (contrast_pairs["pos"] == []) and (contrast_pairs["neg"] == []):
                    args.flag_optimize = False
            elif (i >= args.time1) and (i < args.time2):
                args.flag_dpo_loss = False
                args.flag_attn_loss = True
                args.num_round = 1
            else:
                args.flag_optimize = False

            if args.flag_optimize:
                for i_optimize in range(args.num_optimization):
                    if args.flag_dno:
                        extra_info['input_ids'] = prompt_input_ids
                        extra_info['timesteps'] = t.repeat(latents.shape[0])

                        latents_prev_sample, pred_x0_latents, _, grad_direction, pred_x0_dno_no_grad = optimize_step_dymo(
                            args,
                            latents,
                            pipeline, t,
                            pred_x0_prev,
                            prompt_embeds,
                            do_classifier_free_guidance,
                            cross_attention_kwargs,
                            preference_model_fn,
                            extra_info,
                            generator,
                            contrast_pairs,
                            extra_step_kwargs)

                        pred_x0_prev = pred_x0_latents
                        latents = ddim_step_forward_from_xt_1_to_xt(pipeline.scheduler, t, latents_prev_sample, generator)

                if args.flag_round:
                    for i_optimize in range(args.num_optimization):
                        args.flag_round_stop = False
                        extra_info['input_ids'] = prompt_input_ids
                        extra_info['timesteps'] = t.repeat(latents.shape[0])
                        latents_prev_sample, pred_x0_latents, _, grad_direction, pred_x0_dno_no_grad = optimize_step_dymo(
                            args,
                            latents,
                            pipeline, t,
                            pred_x0_prev,
                            prompt_embeds,
                            do_classifier_free_guidance,
                            cross_attention_kwargs,
                            preference_model_fn,
                            extra_info,
                            generator,
                            contrast_pairs,
                            extra_step_kwargs)

                        pred_x0_prev = pred_x0_latents
                        latents = ddim_step_forward_from_xt_1_to_xt(pipeline.scheduler, t, latents_prev_sample, generator)
                    args.flag_round_stop = True

            if latents_prev_sample is not None:
                latents = latents_prev_sample
            else:
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.cfg_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and args.cfg_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                   guidance_rescale=args.cfg_rescale)
                latents, latent_pred_x0, _ = ddim_step_with_logprob_dymo(pipeline.scheduler, noise_pred, t, latents, **extra_step_kwargs)

                pred_x0_prev = latent_pred_x0

            latent_model_input_ori = torch.cat([latents_ori] * 2) if do_classifier_free_guidance else latents_ori
            latent_model_input_ori = pipeline.scheduler.scale_model_input(latent_model_input_ori, t)
            noise_pred_ori = pipeline.unet(
                latent_model_input_ori,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_uncond_ori, noise_pred_text_ori = noise_pred_ori.chunk(2)
                noise_pred_ori = noise_pred_uncond_ori + args.cfg_scale * (noise_pred_text_ori - noise_pred_uncond_ori)
            if do_classifier_free_guidance and args.cfg_rescale > 0.0:
                noise_pred_ori = rescale_noise_cfg(noise_pred_ori, noise_pred_text_ori, guidance_rescale=args.cfg_rescale)
            latents_ori, _, _ = ddim_step_with_logprob_dymo(pipeline.scheduler, noise_pred_ori, t,
                                                                        latents_ori,
                                                                        variance_noise=None,
                                                                        **extra_step_kwargs)


            if i == len(timesteps) - 1 or ((i + 1) > 0 and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()
                torch.cuda.empty_cache()

    image = pipeline.vae.decode(latents.to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor, return_dict=False,
                                generator=generator, )[0]
    image = pipeline.image_processor.postprocess(image, output_type="pt", do_denormalize=[True] * image.shape[0])

    image_ori = pipeline.vae.decode(latents_ori.to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor, return_dict=False,
                                generator=generator, )[0]
    image_ori = pipeline.image_processor.postprocess(image_ori, output_type="pt", do_denormalize=[True] * image_ori.shape[0])

    image = np.concatenate(((image_ori[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), (image[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)), axis=1)
    image = Image.fromarray(image)
    image.save(save_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_id', default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--prompt',
        default='a photo',
    )
    parser.add_argument(
        '--prompt_json',
        default='./show_prompt_files/prompt1.json',
    )
    parser.add_argument(
        '--cfg_scale',
        default=7.5,
        type=float,
    )
    parser.add_argument(
        '--cfg_rescale',
        default=0.0,
        type=float,
    )
    parser.add_argument(
        '--output_filename',
        default='sdv1-5_RLHF_img.png',
    )
    parser.add_argument(
        '--seed',
        default=40,
        type=int,
    )
    parser.add_argument(
        '--num_timesteps',
        default=50,
        type=int,
    )
    parser.add_argument(
        '--height',
        default=512,
        type=int,
    )
    parser.add_argument(
        '--width',
        default=512,
        type=int,
    )
    parser.add_argument(
        '--num_optimization',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--flag_optimize',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '--opt_stength',
        default=1,
        type=int,
    )
    # Noise Optimize setting
    parser.add_argument(
        '--flag_dno',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '--flag_round',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '--flag_round_stop',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '--num_round',
        default=1,
        type=int,
    )
    parser.add_argument(
        '--flag_dpo_loss',
        default=False,
        type=bool,
    )
    parser.add_argument(
        '--flag_attn_loss',
        default=True,
        type=bool,
    )
    parser.add_argument(
        '--time1',
        default=8,
        type=int,
    )
    parser.add_argument(
        '--time2',
        default=25,
        type=int,
    )

    args = parser.parse_args()
    nlp = spacy.load('en_core_web_sm')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_dtype = torch.float16

    preference_model_func_cfg = dict(
        type="step_aware_preference_model_func",
        model_pretrained_model_name_or_path='yuvalkirstain/PickScore_v1',
        processor_pretrained_model_name_or_path='laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
        ckpt_path='model_ckpts/sd-v1-5_step-aware_preference_model.bin',
        device=device,
        inference_dtype=inference_dtype,
    )
    preference_model_fn = get_preference_model_func(preference_model_func_cfg)

    # loading models
    ckpt_id = args.ckpt_id

    pipeline = StableDiffusionPipeline.from_pretrained(
        ckpt_id,
        torch_dtype=inference_dtype
    )

    # cross-attn
    attn_res = (16, 16)
    pipeline.attention_store = AttentionStore(attn_res)
    attn_procs = {}
    cross_att_count = 0
    for name in pipeline.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue
        cross_att_count += 1
        attn_procs[name] = AttendExciteAttnProcessor(
            attnstore=pipeline.attention_store, place_in_unet=place_in_unet
        )
    pipeline.unet.set_attn_processor(attn_procs)
    pipeline.attention_store.num_att_layers = cross_att_count

    json_file_path = args.prompt_json
    save_path = os.path.join("./results")
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, args.output_filename)
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    args.prompt = json_data["prompt"].lower()
    doc = nlp(args.prompt)
    tokens_dict = find_tokens_from_text(json_data, doc)
    token_p_n_pairs = []
    for i, key in enumerate(tokens_dict):
        if tokens_dict[key] is not None:
            pairs = [[key], tokens_dict[key]]
        else:
            pairs = [[key], None]

        match_list_indices = align_indices(pipeline, args.prompt, pairs)
        result = flatten_lists(match_list_indices)
        token_p_n_pairs.append(result)
    positive_pairs, negative_pairs = process_p_n_groups(token_p_n_pairs)
    p_n_dict = {"pos": positive_pairs, "neg": negative_pairs}

    pipeline.to(device)
    pipeline.text_encoder.requires_grad_(False)

    main(args, args.prompt, save_name, pipeline=pipeline, preference_model_fn=preference_model_fn, contrast_pairs=p_n_dict)

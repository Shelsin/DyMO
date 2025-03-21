import torch

from .builder import PREFERENCE_MODEL_FUNC_BUILDERS
from .models.step_aware_preference_model import StepAwarePreferenceModel

@PREFERENCE_MODEL_FUNC_BUILDERS.register_module(name='step_aware_preference_model_func')
def step_aware_preference_model_func_builder(cfg):
    step_aware_preference_model = StepAwarePreferenceModel(
        model_pretrained_model_name_or_path=cfg["model_pretrained_model_name_or_path"],
        processor_pretrained_model_name_or_path=cfg["processor_pretrained_model_name_or_path"],
        ckpt_path=cfg["ckpt_path"],
        inference_dtype=cfg["inference_dtype"]
    ).eval().to(cfg["device"])
    # step_aware_preference_model.requires_grad_(False)
    for param in step_aware_preference_model.parameters():
        # param.to(cfg["inference_dtype"])
        param.requires_grad = False
    
    # @torch.no_grad()
    def preference_fn(img, extra_info, return_feat = False):
        # b
        scores, images_embeds, text_embeds = step_aware_preference_model.get_preference_score(
            img, 
            extra_info['input_ids'],
            extra_info['timesteps'],
        )
        if return_feat:
            return scores, images_embeds, text_embeds
        else:
            return scores
    
    return preference_fn

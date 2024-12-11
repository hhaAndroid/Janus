# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector
from torch.nn import functional as F


class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        embedding = F.normalize(self.gen_vision_model.quantize.embedding.weight, p=2, dim=-1)
        return self.gen_aligner(embedding[image_ids])
        # return self.gen_aligner(self.gen_embed(image_ids))

    # --------------------------------------------------------------------------------------------------
    def forward(self, input_ids, labels, pixel_values, image_flags, gen_image_bounds, image_id):
        pixel_values_in = pixel_values[0::2, ...]  # b,3,h,w
        pixel_values_out = pixel_values[1::2, ...]  # b,3,h,w

        n = pixel_values_in.shape[0]
        # [b, 576, D]
        vision_model_dtype = next(self.vision_model.parameters()).dtype
        images_embeds_in = self.aligner(self.vision_model(pixel_values_in.to(dtype=vision_model_dtype)))
        images_embeds_in = images_embeds_in[image_flags == 1]

        quant, _, info = self.gen_vision_model(pixel_values_out.to(dtype=vision_model_dtype))
        quant = quant.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        images_embeds_out = self.gen_aligner(quant)
        images_embeds_out = images_embeds_out[image_flags == 1]

        vit_embeds = torch.stack([images_embeds_in, images_embeds_out, images_embeds_out], dim=1).flatten(0, 1)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        B, N, C = input_embeds.shape
        input_ids = input_ids.reshape(B * N)
        input_embeds = input_embeds.reshape(B * N, C)
        selected = (input_ids == image_id)

        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model.model(
            inputs_embeds=input_embeds,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state  # b,c,d
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        start_bound = gen_image_bounds[0::2]
        end_bound = gen_image_bounds[1::2]
        assert len(start_bound) == len(end_bound)

        mask = torch.zeros(hidden_states.shape[0], dtype=torch.bool)
        for start, end in zip(start_bound, end_bound):
            mask[start:end] = True

            # 取出 und_gen 的部分
            # end_image, eos_id, bos_id, start_image
            und_gen = end + 4
            mask[und_gen:und_gen + 576] = True

        assert torch.all(input_ids[mask] == image_id)

        hidden_states_gen = hidden_states[mask]
        assert hidden_states_gen.shape[0] == images_embeds_out.shape[0] * images_embeds_out.shape[1] * 2

        hidden_states_und = hidden_states[~mask]

        labels = labels.reshape(-1)
        und_labels = labels[~mask]
        gen_labels = info[2]
        assert len(gen_labels) == len(labels[mask]) // 2

        gen_logits = self.gen_head(hidden_states_gen)

        # 有条件和无条件加权
        assert gen_logits.shape[0] % 576 == 0

        gen_logits = gen_logits.reshape(-1, 576, gen_logits.shape[-1])
        logit_cond = gen_logits[0::2, ...]
        logit_uncond = gen_logits[1::2, ...]
        cfg_weight = 5
        gen_logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        gen_labels = gen_labels.reshape(-1, 576)

        gen_logits = gen_logits[:, :-1, ...].contiguous()
        gen_labels = gen_labels[:, 1:, ...].contiguous()
        gen_loss = F.cross_entropy(gen_logits.view(-1, gen_logits.shape[-1]).float(), gen_labels.view(-1),
                                   reduction='sum')  # 1, seqlen
        if torch.isnan(gen_loss) and (gen_labels[1:, ...] != -100).sum() == 0:
            # When all labels are -100, the CE loss will return NaN and requires special handling.
            gen_loss = gen_logits.sum() * 0

        und_logits = self.language_model.lm_head(hidden_states_und)
        und_logits = und_logits[:-1, ...].contiguous()
        und_labels = und_labels[1:, ...].contiguous()
        und_loss = F.cross_entropy(und_logits.float(), und_labels, reduction='sum')  # 1, seqlen
        if torch.isnan(und_loss) and (und_labels != -100).sum() == 0:
            # When all labels are -100, the CE loss will return NaN and requires special handling.
            und_loss = und_logits.sum() * 0

        return gen_loss, und_loss
    # --------------------------------------------------------------------------------------------------


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)

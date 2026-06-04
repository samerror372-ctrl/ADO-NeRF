# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import einops
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.utils import save_image

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...configuration_utils import FrozenDict
from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from ...models.lora import adjust_lora_scale_text_encoder
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from .pipeline_output import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker
from src.modules.camera import get_camera_embedding
from src.modules.position_encoding_center import global_position_encoding_3d, get_3d_priors #_o只warp一張 #_center warp+貼
#from src.modules.position_encoding_o import global_position_encoding_3d, get_3d_priors 
from ...schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

import os
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from torchvision.transforms.functional import crop

from PIL import Image
import os


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""


SAVE_INTERVAL = 5 


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    
    
def pack_render_imgs_to_tensor(render_imgs, device, target_hw=None):
    """
    render_imgs: list[dict]，每個元素至少有 'img': Tensor[1,3,H,W]
    target_hw: (H, W)，若指定就 resize 到和 pipeline 一樣的大小
    return: Tensor[F,3,H,W]
    """
    if not render_imgs:
        return None
    imgs = []
    for d in render_imgs:
        im = d["img"]  # Tensor[1,3,H,W]，已是 [-1,1]
        if not torch.is_tensor(im):
            # 萬一不是 tensor（極少數情況），做保險轉換
            im = torch.as_tensor(im)
        im = im.to(device=device, dtype=torch.float32)
        if target_hw is not None and im.shape[-2:] != target_hw:
            im = F.interpolate(im, size=target_hw, mode="bilinear", align_corners=False)
        imgs.append(im)
    return torch.cat(imgs, dim=0)  # [F,3,H,W]

def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionMultiViewPipeline(
    DiffusionPipeline,
    StableDiffusionMixin,
    TextualInversionLoaderMixin,
    LoraLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
            self,
            vae: AutoencoderKL,
            unet: UNet2DConditionModel,
            scheduler,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection = None,
            requires_safety_checker: bool = True,
            zoom_scale: float = 1.0,
            render_imgs = None,
            mask_imgs = None,
            mask_mode = None
            
    ):
        super().__init__()
        self.zoom_scale = zoom_scale
        self.render_imgs = render_imgs
        self.mask_imgs = mask_imgs
        self.mask_mode = mask_mode
        
        
        #breakpoint()
        
        if type(scheduler) == dict:
            import copy
            self.global_scheduler = copy.deepcopy(scheduler)
            scheduler = scheduler[list(scheduler.keys())[0]]
        else:
            self.global_scheduler = None

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            **kwargs,
    ):
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            **kwargs,
        )

        # concatenate for backwards comp
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        return prompt_embeds

    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds

    def prepare_ip_adapter_image_embeds(
            self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                    ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def decode_latents_flatten(self, latents):
        # [B, F, C, H, W] → [B*F, C, H, W]
        B, F, C, H, W = latents.shape
        latents = latents.view(B * F, C, H, W)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image  # shape: [B*F, H, W, 3]


    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
            self,
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            ip_adapter_image=None,
            ip_adapter_image_embeds=None,
            callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if ip_adapter_image is not None and ip_adapter_image_embeds is not None:
            raise ValueError(
                "Provide either `ip_adapter_image` or `ip_adapter_image_embeds`. Cannot leave both `ip_adapter_image` and `ip_adapter_image_embeds` defined."
            )

        if ip_adapter_image_embeds is not None:
            if not isinstance(ip_adapter_image_embeds, list):
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be of type `list` but is {type(ip_adapter_image_embeds)}"
                )
            elif ip_adapter_image_embeds[0].ndim not in [3, 4]:
                raise ValueError(
                    f"`ip_adapter_image_embeds` has to be a list of 3D or 4D tensors but is {ip_adapter_image_embeds[0].ndim}D"
                )

    def prepare_latents(self, batch_size, nframe, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            nframe,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    ## Encode image latent
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        #breakpoint()
        
        return image_latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(    #####  Denoise
            self,
            # prompt: Union[str, List[str]] = None,
            images: PipelineImageInput = None,
            nframe: Optional[int] = 8,
            cond_num: Optional[int] = 1,
            height: Optional[int] = None,
            width: Optional[int] = None,
            intrinsics: Optional[torch.Tensor] = None,
            extrinsics: Optional[torch.Tensor] = None,
            num_inference_steps: int = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        tar_idx_batch = kwargs.get("tar_idx_batch", None) 
        
        #breakpoint()
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        prompt = ""
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        batch_size = 1

        device = self._execution_device
        dtype = torch.float32

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        ### encode cameras ###
        no_camera_emb = kwargs['config'].model_cfg.get("no_camera_emb", False)
        if no_camera_emb:
            camera_embedding = None
        else: ##Plucker
            camera_embedding = get_camera_embedding(intrinsics, extrinsics,
                                                    batch_size, nframe, height, width,
                                                    config=kwargs['config']).to(device=device)
        ### build masks (random visible frames), valid:0, mask:1 ###
        masks = torch.ones((batch_size, nframe, 1, height, width), device=device, dtype=dtype)
        latent_masks = torch.ones((batch_size, nframe, 1, height // self.vae_scale_factor, width // self.vae_scale_factor),
                                  device=device, dtype=dtype)
        masks[:, :cond_num] = 0
        latent_masks[:, :cond_num] = 0
        
        import torch.nn.functional as F
        
        
        ### image latents ###
        # images:[f,3,h,w]
        assert images.ndim == 4
        assert type(images) == torch.Tensor
        images = images.to(dtype=torch.float32, device=device) #[40, 3, 600, 800]
        # Reference Input Latent 把 image 丟進encoder轉為latent
        
        
        # image -> 
        #assert images.shape[0] == 40
        nframe_actual = images.shape[0]
        
        print(f"[Info] Loaded {nframe_actual} images (ref + tar)")

        
        ## images [40, 3, 304, 464]
        #breakpoint()
        
        """
        # 拆出前後 20 張
        ref_images = images[:20]  # [20, 3, H, W]
        B, C, H, W = ref_images.shape

        # 後 20 張為純高斯雜訊
        noise_images = torch.randn((20, C, H, W), dtype=torch.float32, device=device)
        import torch.nn.functional as F

        # 將前 20 張縮小 0.5 倍
        scaled_images = F.interpolate(ref_images, scale_factor=0.5, mode='bilinear', align_corners=False)  # [20, 3, H//2, W//2]

        # 將縮小圖貼到高斯雜訊圖中央
        h_small, w_small = scaled_images.shape[-2:]
        top = (H - h_small) // 2
        left = (W - w_small) // 2
        noise_images[:, :, top:top+h_small, left:left+w_small] = scaled_images

        # 合併成新的 images
        images = torch.cat([ref_images, noise_images], dim=0)  # [40, 3, H, W]
        """     
                
        # 原丟 encoder
        image_latents = self._encode_vae_image(image=images, generator=generator) 
        if image_latents.ndim == 4:
            image_latents = image_latents.unsqueeze(0)
        # image_latents = [1, 32, 4, 75, 100] images_latent的維度


        
        # 建立資料夾
        os.makedirs("Vis/debug_images", exist_ok=True)
        import torch.nn.functional as F

        # images shape: [32, 3, H, W]
        for idx, img in enumerate(images):
            vutils.save_image(img, f"Vis/debug_images/img_{idx:02d}.png", normalize=True)
            # 464, 304
        
        # 處理resize image
        """
        # 建立資料夾
        os.makedirs("debug_images/resize", exist_ok=True)
        """
        
        # 前 20 張圖片
        
        # 自動取前一半張圖片
        num_images = images.shape[0]   # B
        half = num_images // 2
        ref_images = images[:half]     # [B//2, C, H, W]

        #ref_images = images[:20]  # [20, 3, H, W]
        #breakpoint()
        B, C, H, W = ref_images.shape

        # 使用 interpolate 縮小
        #### zoom_scale
        
        resized_images = F.interpolate(ref_images, scale_factor=self.zoom_scale, mode='bilinear', align_corners=False)  # [20, 3, H//2, W//2]

        # 存檔
        for idx, img in enumerate(resized_images):
            vutils.save_image(img, f"Vis/debug_images/resize/img_{idx:02d}.png", normalize=True)
        
        
        
        # 丟入 VAE encoder
        resize_image_latents = self._encode_vae_image(resized_images, generator=generator)  # [20, 4, h', w']

        # breakpoint()
        
        
        

        # 和你現有邏輯對齊
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor
        device = self._execution_device

        # 1) 打包 GT 影像成 [F,3,H,W]
        render_batch = pack_render_imgs_to_tensor(self.render_imgs, device=device, target_hw=(height, width))  # [F,3,H,W] 或 None
        mask_batch = pack_render_imgs_to_tensor(self.mask_imgs, device=device, target_hw=(height, width))

        
        # 2) 轉 latent（同 images → image_latents 的方式）
        if render_batch is not None:
            render_latents = self._encode_vae_image(image=render_batch, generator=generator)  # [F,4,h',w']
            if render_latents.ndim == 4:
                render_latents = render_latents.unsqueeze(0)  # -> [1,F,4,h',w']
        else:
            render_latents = None

        # === 將 mask 以 max pooling 縮到與 latent 相同的 h', w'，並存圖 ===
        """
        
        """
        #breakpoint()

        if (render_latents is not None) and (mask_batch is not None):
            # render_latents: [1, F, 4, h', w']
            _, F_, _, h_small, w_small = render_latents.shape

            # ---- 單通道 ----
            if mask_batch.ndim != 4:
                raise ValueError(f"mask_batch 預期為 [F,C,H,W]，但拿到 {mask_batch.shape}")
            mask_1c = mask_batch if mask_batch.shape[1] == 1 else mask_batch[:, :1, :, :]

            # ---- 對齊到 (h', w') ----
            #   用 adaptive_max_pool2d 保留 1 區域（你原本的作法）
            mask_pooled = F.adaptive_max_pool2d(mask_1c, output_size=(h_small, w_small))  # [F,1,h',w']

            # ---- 二值化，得到 base（= 35）----
            base = (mask_pooled >= 0.5).float()  # [F,1,h',w']，在同一個 device（CUDA） #0.5

            # ---- 形態學膨脹：用 max-pool 模擬 dilation ----
            # kernel=5x5 與 iterations 對應 dilation 範圍
            def dilate(x, k=5, iters=1):
                pad = k // 2
                out = x
                for _ in range(iters):
                    out = F.max_pool2d(out, kernel_size=k, stride=1, padding=pad)
                return out

            mask_35 = base                                      # 原始
            mask_25 = dilate(base, k=5, iters=1)                # 輕度擴張
            mask_15 = dilate(base, k=5, iters=2)                # 更大擴張

            # ---- 增 batch 維度，對齊 render_latents: [1,F,1,h',w'] ----
            self.mask_latents_35 = mask_35.unsqueeze(0)
            self.mask_latents_25 = mask_25.unsqueeze(0)
            self.mask_latents_15 = mask_15.unsqueeze(0)

            # ---- 儲存可視化（值在 [0,1]，直接存即可）----
            base_dir = os.path.join(getattr(kwargs['config'], "save_path", "."), "mask_multi")
            os.makedirs(base_dir, exist_ok=True)

            def dump_folder(tensor4d, name):
                # tensor4d: [1,F,1,h',w']
                subdir = os.path.join(base_dir, f"mask_{name}")
                os.makedirs(subdir, exist_ok=True)
                for i in range(tensor4d.shape[1]):
                    save_image(tensor4d[0, i], os.path.join(subdir, f"{i:05d}.png"))

            dump_folder(self.mask_latents_35, "35")
            dump_folder(self.mask_latents_25, "25")
            dump_folder(self.mask_latents_15, "15")

        else:
            self.mask_latents_35 = None
            self.mask_latents_25 = None
            self.mask_latents_15 = None
                
                    

                
        # 假設 render_batch: Tensor[F,3,H,W]，值域 [-1,1]
        save_dir = "Vis/vis_render_img"
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for idx, img in enumerate(render_batch):  # img: [3,H,W]
                img01 = (img.clamp(-1, 1) + 1) / 2.0  # [-1,1] -> [0,1]
                vutils.save_image(img01.cpu(), os.path.join(save_dir, f"render_{idx:03d}.png"))
                
        #breakpoint()
        
        
        
        ###  準備 3D priors #各種condition  ## warp
        if camera_embedding is None:
            add_inputs = masks  ##  mask
        else:
            add_inputs = torch.cat([masks, camera_embedding], dim=2)  # [b,f,c,h,w]  ## camera_embedding	
        coords = None
        if kwargs['config'].model_cfg.get("enable_depth", False):
            if kwargs['config'].model_cfg.get("priors3d", False):
                coords = get_3d_priors(kwargs['config'], kwargs["depth"], intrinsics, extrinsics,
                                       cond_num, nframe=nframe, device=device, colors=images, latents=image_latents,
                                       vae=kwargs['vae'], prior_type=kwargs['config'].model_cfg.get("prior_type", "3dpe"), tar_idx_batch=tar_idx_batch, zoom_scale=self.zoom_scale)
                
                save_dir = os.path.join(getattr(kwargs['config'], "save_path", "."), "depth")
                os.makedirs(save_dir, exist_ok=True)
                #import numpy as np
                
                #import os

                depth_tensor = kwargs["depth"].detach().cpu()   # [N,1,H,W]
                N, _, H, W = depth_tensor.shape

                for i in range(N):
                    depth = depth_tensor[i, 0].numpy()

                    # 為了避免過曝，做 min-max normalize
                    d_min, d_max = np.percentile(depth, 1), np.percentile(depth, 99)
                    depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)

                    depth_img = (depth_norm * 255).astype(np.uint8)
                    save_path = os.path.join(save_dir, f"depth_{i:03d}.png")
                    cv2.imwrite(save_path, depth_img)    
                
                #breakpoint()
            else:
                depth = einops.rearrange(kwargs["depth"], "(b f) c h w -> b f c h w",
                                         b=batch_size, f=nframe).to(device=device)
                depth[:, cond_num:] = 0
                add_inputs = torch.cat([add_inputs, depth], dim=2)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            if camera_embedding is None:
                negative_inputs = masks
            else:
                negative_inputs = torch.cat([masks, torch.zeros_like(camera_embedding)], dim=2)
            if kwargs['config'].model_cfg.get("priors3d", False):
                if type(coords) == list:
                    for j in range(len(coords)):
                        coords[j] = torch.cat([torch.zeros_like(coords[j]), coords[j]], dim=0)
                else:
                    coords = torch.cat([torch.zeros_like(coords), coords], dim=0)
            elif kwargs['config'].model_cfg.get("enable_depth", False):
                negative_inputs = torch.cat([negative_inputs, torch.zeros_like(depth)], dim=2)
            add_inputs = torch.cat([negative_inputs, add_inputs], dim=0)

        #breakpoint()
        
        
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        if self.global_scheduler is not None and nframe - cond_num in self.global_scheduler:
            scheduler = self.global_scheduler[nframe - cond_num]
        else:
            scheduler = self.scheduler

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables  
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            nframe - cond_num,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        latents = torch.cat([image_latents[:, :cond_num], latents], dim=1)
        
        #breakpoint()
        # 
        #測試1 直接用黑圖產生latent呢？ 爛掉
        # latents = image_latents
        
        #測試2 直接把原本的圖片重複兩次
        #latents = torch.cat([image_latents[:, :cond_num], image_latents[:, :cond_num]], dim=1)
        
        
        #####
        #breakpoint()
        

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        ### get domain switcher ###
        domain_dict = kwargs['config'].model_cfg.get("domain_dict", None)
        if domain_dict is not None:
            tags = kwargs["tag"][::nframe]
            tags = tags[::kwargs['config'].nframe]
            class_labels = [domain_dict.get(tag, domain_dict['others']) for tag in tags]
            class_labels = torch.tensor(class_labels, dtype=torch.long, device=device)
        else:
            class_labels = None

        if kwargs.get("class_label", None) is not None and class_labels is not None:
            class_labels = torch.ones_like(class_labels) * kwargs.get("class_label", None)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        self._num_timesteps = len(timesteps)
        
        # ===== 迴圈外做一次 =====
        timesteps_idx = [int(x) for x in timesteps.detach().to("cpu").view(-1).tolist()]
        alphas = scheduler.alphas_cumprod  # CPU tensor
        N_alpha = alphas.shape[0]
                
        L = len(timesteps)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):   ### 
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                t_idx      = timesteps_idx[i]
                t_prev_idx = timesteps_idx[i + 1] if i < L - 1 else t_idx
                
                # 給 UNet/scheduler 的 timestep（device tensor）
                t_tensor      = torch.tensor(t_idx,      dtype=torch.long, device=latents.device)
                t_prev_tensor = torch.tensor(t_prev_idx, dtype=torch.long, device=latents.device)

                # 當前latent
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                # latents = [1, 40, 4, 51, 77]
                # latent_model_input = [2, 40, 4, 51, 77]
                # breakpoint()
                
                # predict the noise residual  
                noise_pred = self.unet(     ## [80, 4, 51, 77] 
                    latent_model_input,  
                    t_tensor,  
                    add_inputs=add_inputs,
                    encoder_hidden_states=None,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    class_labels=class_labels,
                    added_cond_kwargs=added_cond_kwargs,
                    coords=coords, #warp
                    return_dict=False,
                    cond_num=cond_num,
                    key_rescale=kwargs.get("key_rescale", None)
                )[0]
                #breakpoint()
                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
               
               
                # noise_pred  [40, 4, 51, 77]
                # resize_image_latents [20, 4, 25, 38]
                # compute the previous noisy sample x_t -> x_t-1
                
                #### !!! Scheduler !!! ####   #####  DDIM 的 denoising 公式
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]  ## 
                
                #breakpoint()
                ##### dx = (xₜ − x̂₀)/σₜ  藏在scheduler
                # [1, 40, 4, 51, 77]
                
                ######  把 ref latent 撒上noise貼到中間  ######
                # resize_image_latents: [20, 4, 25, 38]
                B, F, C, H, W = latents.shape  # [1, 40, 4, 51, 77]
                resize_H, resize_W = resize_image_latents.shape[-2:]

                # 取得當前 timestep 對應的 noise scale（σₜ）
                #sigma_t = scheduler.sigmas[i]  # i 是目前的 timestep index
                #t = timesteps[i]

                if hasattr(scheduler, "sigmas"):
                    sigma_t = scheduler.sigmas[i]
                else:
                    # For schedulers like DDIM
                    alpha_cumprod_t = scheduler.alphas_cumprod[t]
                    #alpha_cumprod_prev = scheduler.alphas_cumprod[t-20]
                    #breakpoint()
                    sigma_t = (1 - alpha_cumprod_t).sqrt()

                noise = torch.randn_like(resize_image_latents) * sigma_t.to(resize_image_latents.device)
                # 分段貼 latent：根據 timestep 決定是否貼入縮小的 ref latent
                
                replace_with = None
                
                
                ## 貼clean latent
                ## 正確 noise
                
            # —— 每一步都用 35 的 mask 進行 latent blend ——
                if render_latents is not None:
                    B, F, C, H, W = latents.shape
                    device = latents.device
                    dtype  = latents.dtype

                    # 固定使用 35 的 mask
                    mask_name = "mask_latents_35"
                    mask_lat = getattr(self, mask_name, None)

                    # 若沒有對應 mask 就跳過
                    use_mask_blend = mask_lat is not None
                    if not use_mask_blend:
                        print(f"[Warn] {mask_name} is None → skip latent blend at i={i}")
                    else:
                        # === 準備 target 區段（從 cond_num 到最後）===
                        tgt_idx = torch.arange(cond_num, F, device=device)
                        replace_n = tgt_idx.numel()

                        # === 準備 render_latents ===
                        rdl = render_latents
                        if rdl.ndim == 4:
                            rdl = rdl.unsqueeze(0)  # [1, F_rd, 4, h', w']
                        rdl = rdl.to(device=device, dtype=dtype)
                        F_rd = rdl.shape[1]
                        if F_rd < replace_n:
                            reps = (replace_n + F_rd - 1) // F_rd
                            rdl = rdl.repeat(1, reps, 1, 1, 1)
                        rdl = rdl[:, :replace_n]

                        # === 根據排程把 noise 注入到 render latent（用上一時間步強度）===
                        if i < L - 1:
                            t_prev = timesteps[i + 1]  # 重要：前一步的強度
                        else:
                            t_prev = timesteps[i]

                        if hasattr(scheduler, "alphas_cumprod"):
                            alpha_bar_prev = scheduler.alphas_cumprod[t_prev_idx].to(device=device, dtype=dtype)
                            sqrt_ab  = alpha_bar_prev.sqrt()
                            sqrt_omb = (1.0 - alpha_bar_prev).sqrt()
                            noise = torch.randn_like(rdl)
                            rdl_noisy = sqrt_ab * rdl + sqrt_omb * noise
                        elif hasattr(scheduler, "sigmas"):
                            sigma_prev = scheduler.sigmas[i + 1] if i < L - 1 else scheduler.sigmas[i]
                            noise = torch.randn_like(rdl)
                            rdl_noisy = rdl + sigma_prev.to(device=device, dtype=dtype) * noise
                        else:
                            raise RuntimeError("Unknown scheduler type: neither alphas_cumprod nor sigmas are available.")

                        # === Mask blend ===
                        mask_tgt = mask_lat.clamp_(0, 1)  # 白(1)→保留原本 latent；黑(0)→用 rdl_noisy
                        _, _, _, h_small, w_small = rdl.shape
                        if rdl_noisy.shape[2] != latents.shape[2]:
                            raise ValueError(f"Channel mismatch: rdl_noisy C={rdl_noisy.shape[2]} vs latents C={latents.shape[2]}")

                        mask_tgt     = mask_tgt.expand(B, replace_n, latents.shape[2], h_small, w_small)
                        rdl_noisy_b  = rdl_noisy.expand(B, replace_n, -1, -1, -1)
                        lat_tgt_orig = latents[:, tgt_idx]
                        lat_tgt_blend = mask_tgt * lat_tgt_orig + (1.0 - mask_tgt) * rdl_noisy_b
                        latents[:, tgt_idx] = lat_tgt_blend

                        print(f"✨ Latent blend at step {i}/{L} using {mask_name}")

                        # ===== Noise Resampling（同一時間步 t 的重抽樣），只對 target 區段 =====
                        alpha_bar_t    = scheduler.alphas_cumprod[t_idx].to(device=latents.device, dtype=latents.dtype)
                        alpha_t        = alpha_bar_t.sqrt()
                        sigma_t        = (1.0 - alpha_bar_t).sqrt()

                        alpha_bar_prev = scheduler.alphas_cumprod[t_prev_idx].to(device=latents.device, dtype=latents.dtype)
                        alpha_prev     = alpha_bar_prev.sqrt()
                        sigma_prev     = (1.0 - alpha_bar_prev).sqrt()

                        pred_type = getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon")

                        extra_resample_steps = 3  # 需要就調整次數
                        for _ in range(extra_resample_steps):
                            # (A) 用 s = t_prev 推出 x0（僅 target 區段）
                            latent_in_prev = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                            latent_in_prev = scheduler.scale_model_input(latent_in_prev, t_prev_tensor)

                            noise_pred_prev = self.unet(
                                latent_in_prev, t_prev_tensor,
                                add_inputs=add_inputs,
                                encoder_hidden_states=None,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                class_labels=class_labels,
                                added_cond_kwargs=added_cond_kwargs,
                                coords=coords,
                                return_dict=False,
                                cond_num=cond_num,
                                key_rescale=kwargs.get("key_rescale", None)
                            )[0]
                            if self.do_classifier_free_guidance:
                                n_u, n_t = noise_pred_prev.chunk(2)
                                noise_pred_prev = n_u + self.guidance_scale * (n_t - n_u)
                                if self.guidance_rescale > 0.0:
                                    noise_pred_prev = rescale_noise_cfg(noise_pred_prev, n_t, guidance_rescale=self.guidance_rescale)

                            eps_full   = noise_pred_prev if noise_pred_prev.dim() == 5 else noise_pred_prev.unsqueeze(0)
                            x_s_region = latents[:, tgt_idx]    # [B, replace_n, 4, H, W]
                            eps_region = eps_full[:, tgt_idx]   # [B, replace_n, 4, H, W]
                            if eps_region.shape[2] != x_s_region.shape[2]:
                                eps_region = eps_region[:, :, :x_s_region.shape[2]]

                            if pred_type == "epsilon":
                                x0_region = (x_s_region - sigma_prev * eps_region) / alpha_prev
                            elif pred_type == "v_prediction":
                                v = eps_region
                                x0_region = alpha_prev * x_s_region - sigma_prev * v
                            elif pred_type == "sample":
                                x0_region = eps_region
                            else:
                                raise ValueError(f"Unknown prediction_type: {pred_type}")

                            # (B) 以「當前步 t」把 x0 升噪到 x_t（僅 target 區段）
                            eps_t      = torch.randn_like(x0_region)
                            x_t_region = alpha_t * x0_region + sigma_t * eps_t

                            # (C) 以同一個 t 做一步 denoise，只回寫 target 區段
                            latents_xt = latents.detach().clone()
                            latents_xt[:, tgt_idx] = x_t_region

                            latent_in_t = torch.cat([latents_xt] * 2) if self.do_classifier_free_guidance else latents_xt
                            latent_in_t = scheduler.scale_model_input(latent_in_t, t_tensor)

                            noise_pred_t = self.unet(
                                latent_in_t, t_tensor,
                                add_inputs=add_inputs,
                                encoder_hidden_states=None,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                class_labels=class_labels,
                                added_cond_kwargs=added_cond_kwargs,
                                coords=coords,
                                return_dict=False,
                                cond_num=cond_num,
                                key_rescale=kwargs.get("key_rescale", None)
                            )[0]
                            if self.do_classifier_free_guidance:
                                n_u, n_t = noise_pred_t.chunk(2)
                                noise_pred_t = n_u + self.guidance_scale * (n_t - n_u)
                                if self.guidance_rescale > 0.0:
                                    noise_pred_t = rescale_noise_cfg(noise_pred_t, n_t, guidance_rescale=self.guidance_rescale)

                            latents_step = scheduler.step(
                                noise_pred_t, t_tensor, latents_xt, **extra_step_kwargs, return_dict=False
                            )[0]

                            # 只更新 target 區段；條件幀鎖回
                            latents[:, tgt_idx]  = latents_step[:, tgt_idx]
                            latents[:, :cond_num] = image_latents[:, :cond_num]

                        # 每輪最後再次鎖回條件幀，保持穩定
                        latents[:, :cond_num] = image_latents[:, :cond_num]

                # 確保條件幀在本步結尾維持不變
                latents[:, :cond_num] = image_latents[:, :cond_num]


                # # ===================================================
                # # B. 新增的 49 步：把「多張 cond」縮小後一一貼到 target，
                # #    並且貼進去的區塊用 (i+1) 對應的 noise 強度重抽
                # # ===================================================
                # extra_paste_steps = {49}
                # if i in extra_paste_steps:
                #     zoom = float(getattr(self, "zoom_scale", 1.0))
                #     if zoom <= 0:
                #         raise ValueError(f"zoom_scale must be > 0, got {zoom}")

                #     device = latents.device
                #     dtype  = latents.dtype

                #     # 目標區段：cond_num ~ F-1
                #     tgt_idx = torch.arange(cond_num, F, device=device)

                #     # 最後 latent 的空間大小
                #     _, _, _, H, W = latents.shape
                #     import torch.nn.functional as Fnn

                #     new_h = max(1, int(H * zoom))
                #     new_w = max(1, int(W * zoom))
                #     top  = (H - new_h) // 2
                #     left = (W - new_w) // 2

                #     # === 取得「要對齊的噪聲強度」：用 t_prev_idx (也就是 i+1 那一步) ===
                #     # 因為上面你已經做完 scheduler.step(...)，現在 latents 是 x_{t_prev}，
                #     # 所以貼進去的 patch 也要長得像 x_{t_prev}
                #     use_alpha = False
                #     if hasattr(scheduler, "alphas_cumprod"):
                #         alpha_bar_prev = scheduler.alphas_cumprod[t_prev_idx].to(device=device, dtype=dtype)
                #         sqrt_ab_prev   = alpha_bar_prev.sqrt()
                #         sqrt_omb_prev  = (1.0 - alpha_bar_prev).sqrt()
                #         use_alpha = True
                #     elif hasattr(scheduler, "sigmas"):
                #         # e.g. euler / dpm++ 等
                #         # i < L-1 就用下一個 sigma，否則就用當前
                #         sigma_prev = scheduler.sigmas[i + 1] if i < L - 1 else scheduler.sigmas[i]
                #         sigma_prev = sigma_prev.to(device=device, dtype=dtype)
                #         use_alpha = False
                #     else:
                #         raise RuntimeError("Unknown scheduler type: neither alphas_cumprod nor sigmas")

                #     # 一張一張 target 處理
                #     for k, fidx in enumerate(tgt_idx):
                #         # 這張 target 要用第幾張 cond
                #         src_idx = k % cond_num   # ← 多張對應前面cond第幾張
                #         ref_lat = image_latents[:, src_idx:src_idx+1].squeeze(1)   # [B, C, H, W]

                #         # 1) 先縮小
                #         if zoom != 1.0:
                #             ref_lat_resized = Fnn.interpolate(
                #                 ref_lat,
                #                 size=(new_h, new_w),
                #                 mode="bilinear",
                #                 align_corners=False,
                #             )
                #         else:
                #             ref_lat_resized = ref_lat

                #         # 2) 再把這塊「升噪」到 t_prev level
                #         if use_alpha:
                #             # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0 + sqrt(1 - alpha_bar_{t-1}) * eps
                #             noise_patch = torch.randn_like(ref_lat_resized)
                #             ref_lat_noisy = sqrt_ab_prev * ref_lat_resized + sqrt_omb_prev * noise_patch
                #         else:
                #             # x_{t-1} = x0 + sigma_{t-1} * eps
                #             noise_patch = torch.randn_like(ref_lat_resized)
                #             ref_lat_noisy = ref_lat_resized + sigma_prev * noise_patch

                #         # 3) 貼回去 target frame 的中間
                #         frame_lat = latents[:, fidx]  # [B, C, H, W]
                #         frame_lat[:, :, top:top+new_h, left:left+new_w] = ref_lat_noisy
                #         latents[:, fidx] = frame_lat

                #     print(f"🟣 Extra zoom+re-noise paste at step {i}: zoom={zoom}, targets={tgt_idx.tolist()}")

                        # breakpoint()
                
                ### blending ###
                # 每一步都把原ref latent蓋過去（不信任denoise model認為的而是信任原始的image)
                latents[:, :cond_num] = image_latents[:, :cond_num]
            
                if i > 0 or i == len(timesteps) - 1:
                    
                    # decode 目前的 latent
                    imgs = self.decode_latents_flatten(latents)

                    # 存圖（每張圖存一張）
                    for b_idx, img in enumerate(imgs):
                        save_path = f"Vis/denoise_vis/office_2/noiset-1/step_{i:03d}/b{b_idx}.png"
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        Image.fromarray((img * 255).astype("uint8")).save(save_path)
                            
                                    
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(scheduler, "order", 1)
                        callback(step_idx, t, latents)
                        
            ### for end

            

        
        if latents.ndim == 5:
            latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")

        if not output_type == "latent":
            sub_size = 8
            slice_num = latents.shape[0] // sub_size
            if latents.shape[0] % sub_size != 0:
                slice_num += 1
            image = []
            for i in range(slice_num):
                image_ = self.vae.decode(latents[i * sub_size:(i + 1) * sub_size] / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                image.append(image_)
            image = torch.cat(image, dim=0)
            # image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
        else:
            image = latents
        has_nsfw_concept = None

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

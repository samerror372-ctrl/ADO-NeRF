import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import einops
import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

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
from src.modules.position_encoding_center import global_position_encoding_3d, get_3d_priors 
from ...schedulers.scheduling_ddim_inverse import DDIMInverseScheduler

import os
import torch.nn.functional as F
import numpy as np

logger = logging.get_logger(__name__) 

EXAMPLE_DOC_STRING = ""

SAVE_INTERVAL = 5 

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
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
    if not render_imgs:
        return None
    imgs = []
    for d in render_imgs:
        im = d["img"] 
        if not torch.is_tensor(im):
            im = torch.as_tensor(im)
        im = im.to(device=device, dtype=torch.float32)
        if target_hw is not None and im.shape[-2:] != target_hw:
            im = F.interpolate(im, size=target_hw, mode="bilinear", align_corners=False)
        imgs.append(im)
    return torch.cat(imgs, dim=0) 

def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
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
        
        if type(scheduler) == dict:
            import copy
            self.global_scheduler = copy.deepcopy(scheduler)
            scheduler = scheduler[list(scheduler.keys())[0]]
        else:
            self.global_scheduler = None

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
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
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
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
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

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
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if self.text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
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
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    def decode_latents_flatten(self, latents):
        B, F, C, H, W = latents.shape
        latents = latents.view(B * F, C, H, W)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image  

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

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

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_guidance_scale_embedding(
            self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
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

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        # ==========================================================
        # 🛡️ VAE 溢出终极防线：强制挂起 autocast，全程使用 FP32 编码
        # ==========================================================
        with torch.autocast("cuda", enabled=False):
            # 强行拉回单精度，防止出现 NaN
            image = image.to(dtype=torch.float32)
            self.vae.to(dtype=torch.float32)
            
            if isinstance(generator, list):
                image_latents = [
                    retrieve_latents(self.vae.encode(image[i: i + 1]), generator=generator[i])
                    for i in range(image.shape[0])
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        # 乘以缩放因子
        image_latents = self.vae.config.scaling_factor * image_latents
        
        # 编码安全结束后，转回 UNet 所需的半精度 (float16) 以节省显存
        return image_latents.to(dtype=self.unet.dtype)

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            images: PipelineImageInput = None,
            pred_rgb: Optional[torch.Tensor] = None,   # [新增] 预测目标视角RGB
            blur_mask: Optional[torch.Tensor] = None,  # [新增] 连续平滑模糊Mask
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
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        tar_idx_batch = kwargs.get("tar_idx_batch", None) 
        
        if callback is not None:
            deprecate("callback", "1.0.0", "Passing `callback` is deprecated, use `callback_on_step_end`")
        if callback_steps is not None:
            deprecate("callback_steps", "1.0.0", "Passing `callback_steps` is deprecated, use `callback_on_step_end`")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        prompt = ""
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt,
            prompt_embeds, negative_prompt_embeds, ip_adapter_image,
            ip_adapter_image_embeds, callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        batch_size = 1
        device = self._execution_device
        dtype = torch.float32

        no_camera_emb = kwargs.get('config', {}).model_cfg.get("no_camera_emb", False) if 'config' in kwargs else False
        if no_camera_emb:
            camera_embedding = None
        else:
            camera_embedding = get_camera_embedding(
                intrinsics, extrinsics, batch_size, nframe, height, width, 
                config=kwargs.get('config', None)
            ).to(device=device)

        masks = torch.ones((batch_size, nframe, 1, height, width), device=device, dtype=dtype)
        masks[:, :cond_num] = 0
        
        import torch.nn.functional as F
        
        assert images.ndim == 4 and isinstance(images, torch.Tensor)
        images = images.to(dtype=torch.float32, device=device)
        
        image_latents = self._encode_vae_image(image=images, generator=generator) 
        if image_latents.ndim == 4:
            image_latents = image_latents.unsqueeze(0)

        # [修改] 精确提取源视角，取代写死的 half
        ref_images = images[:cond_num] 
        resized_images = F.interpolate(ref_images, scale_factor=self.zoom_scale, mode='bilinear', align_corners=False)
        resize_image_latents = self._encode_vae_image(resized_images, generator=generator)

        # [修改] 动态处理 pred_rgb
        if pred_rgb is not None:
            pred_rgb = pred_rgb.to(device=device, dtype=dtype)
            if pred_rgb.shape[-2:] != (height, width):
                pred_rgb = F.interpolate(pred_rgb, size=(height, width), mode="bilinear", align_corners=False)
            render_latents = self._encode_vae_image(image=pred_rgb, generator=generator)
            if render_latents.ndim == 4:
                render_latents = render_latents.unsqueeze(0)
        else:
            render_latents = None

        # [修改] 动态处理连续 blur_mask，使用 soft dilate
        if render_latents is not None and blur_mask is not None:
            blur_mask = blur_mask.to(device=device, dtype=dtype)
            _, F_, _, h_small, w_small = render_latents.shape
            
            mask_pooled = F.interpolate(blur_mask, size=(h_small, w_small), mode='area')
            base = mask_pooled.clamp(0.0, 1.0)

            def soft_dilate(x, k=5, iters=1):
                pad = k // 2
                out = x
                for _ in range(iters):
                    out = F.max_pool2d(out, kernel_size=k, stride=1, padding=pad)
                return out

            self.mask_latents_35 = base.unsqueeze(0)
            self.mask_latents_25 = soft_dilate(base, k=5, iters=1).unsqueeze(0)
            self.mask_latents_15 = soft_dilate(base, k=5, iters=2).unsqueeze(0)
        else:
            self.mask_latents_35 = self.mask_latents_25 = self.mask_latents_15 = None

        # [修改] 深度自动 Padding
        depth_in = kwargs.get("depth", None)
        if depth_in is not None:
            depth_in = depth_in.to(device=device, dtype=dtype)
            if depth_in.shape[0] == cond_num:
                target_num = nframe - cond_num
                zero_depths = torch.zeros((target_num, depth_in.shape[1], depth_in.shape[2], depth_in.shape[3]), device=device, dtype=dtype)
                depth_in = torch.cat([depth_in, zero_depths], dim=0)
            kwargs["depth"] = depth_in

        if camera_embedding is None:
            add_inputs = masks
        else:
            add_inputs = torch.cat([masks, camera_embedding], dim=2)
            
        coords = None
        if kwargs.get('config') and kwargs['config'].model_cfg.get("enable_depth", False):
            if kwargs['config'].model_cfg.get("priors3d", False):
                coords = get_3d_priors(
                    kwargs['config'], kwargs["depth"], intrinsics, extrinsics,
                    cond_num, nframe=nframe, device=device, colors=images, latents=image_latents,
                    vae=kwargs.get('vae', self.vae), prior_type=kwargs['config'].model_cfg.get("prior_type", "3dpe"), 
                    tar_idx_batch=tar_idx_batch, zoom_scale=self.zoom_scale
                )
            else:
                depth_concat = einops.rearrange(kwargs["depth"], "(b f) c h w -> b f c h w", b=batch_size, f=nframe).to(device=device)
                depth_concat[:, cond_num:] = 0
                add_inputs = torch.cat([add_inputs, depth_concat], dim=2)

        if self.do_classifier_free_guidance:
            if camera_embedding is None:
                negative_inputs = masks
            else:
                negative_inputs = torch.cat([masks, torch.zeros_like(camera_embedding)], dim=2)
            
            if kwargs.get('config') and kwargs['config'].model_cfg.get("priors3d", False) and coords is not None:
                if isinstance(coords, list):
                    for j in range(len(coords)):
                        coords[j] = torch.cat([torch.zeros_like(coords[j]), coords[j]], dim=0)
                else:
                    coords = torch.cat([torch.zeros_like(coords), coords], dim=0)
            elif kwargs.get('config') and kwargs['config'].model_cfg.get("enable_depth", False):
                negative_inputs = torch.cat([negative_inputs, torch.zeros_like(depth_concat)], dim=2)
            add_inputs = torch.cat([negative_inputs, add_inputs], dim=0)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, ip_adapter_image_embeds, device,
                batch_size * num_images_per_prompt, self.do_classifier_free_guidance,
            )

        scheduler = self.global_scheduler[nframe - cond_num] if (self.global_scheduler is not None and nframe - cond_num in self.global_scheduler) else self.scheduler
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, timesteps, sigmas)

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, nframe - cond_num,
            num_channels_latents, height, width, dtype, device, generator, latents,
        )
        latents = torch.cat([image_latents[:, :cond_num], latents], dim=1)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {"image_embeds": image_embeds} if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) else None

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        domain_dict = kwargs.get('config', {}).model_cfg.get("domain_dict", None) if kwargs.get('config') else None
        if domain_dict is not None:
            tags = kwargs["tag"][::nframe][::kwargs['config'].nframe]
            class_labels = torch.tensor([domain_dict.get(tag, domain_dict['others']) for tag in tags], dtype=torch.long, device=device)
        else:
            class_labels = None

        if kwargs.get("class_label", None) is not None and class_labels is not None:
            class_labels = torch.ones_like(class_labels) * kwargs.get("class_label", None)

        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        self._num_timesteps = len(timesteps)
        timesteps_idx = [int(x) for x in timesteps.detach().to("cpu").view(-1).tolist()]
        L = len(timesteps)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt: continue

                t_idx = timesteps_idx[i]
                t_prev_idx = timesteps_idx[i + 1] if i < L - 1 else t_idx
                t_tensor = torch.tensor(t_idx, dtype=torch.long, device=latents.device)
                t_prev_tensor = torch.tensor(t_prev_idx, dtype=torch.long, device=latents.device)

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                noise_pred = self.unet(
                    latent_model_input, t_tensor, add_inputs=add_inputs,
                    encoder_hidden_states=None, timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    class_labels=class_labels, added_cond_kwargs=added_cond_kwargs,
                    coords=coords, return_dict=False, cond_num=cond_num,
                    key_rescale=kwargs.get("key_rescale", None)
                )[0]
                
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if self.guidance_rescale > 0.0:
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
               
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                trigger_points = {
                    int(0.3 * L): "mask_latents_15",
                    int(0.5 * L): "mask_latents_25",
                    int(0.7 * L): "mask_latents_35",
                }

                if (render_latents is not None) and (i in trigger_points):
                    B, F, C, H, W = latents.shape
                    mask_name = trigger_points[i]
                    mask_lat = getattr(self, mask_name, None)

                    if mask_lat is not None:
                        tgt_idx = torch.arange(cond_num, F, device=device)
                        replace_n = tgt_idx.numel()
                        rdl = render_latents.to(device=device, dtype=latents.dtype)
                        if rdl.shape[1] < replace_n:
                            rdl = rdl.repeat(1, (replace_n + rdl.shape[1] - 1) // rdl.shape[1], 1, 1, 1)
                        rdl = rdl[:, :replace_n]

                        t_prev_val = timesteps[i + 1] if i < L - 1 else timesteps[i]
                        if hasattr(scheduler, "alphas_cumprod"):
                            alpha_bar_prev = scheduler.alphas_cumprod[t_prev_idx].to(device=device)
                            rdl_noisy = alpha_bar_prev.sqrt() * rdl + (1.0 - alpha_bar_prev).sqrt() * torch.randn_like(rdl)
                        else:
                            sigma_prev = scheduler.sigmas[i + 1] if i < L - 1 else scheduler.sigmas[i]
                            rdl_noisy = rdl + sigma_prev.to(device=device) * torch.randn_like(rdl)

                        mask_tgt = mask_lat.clamp(0, 1).expand(B, replace_n, C, rdl.shape[3], rdl.shape[4])
                        latents[:, tgt_idx] = mask_tgt * latents[:, tgt_idx] + (1.0 - mask_tgt) * rdl_noisy

                        alpha_bar_t = scheduler.alphas_cumprod[t_idx].to(device=device)
                        alpha_bar_prev = scheduler.alphas_cumprod[t_prev_idx].to(device=device)
                        pred_type = getattr(scheduler.config, "prediction_type", "epsilon")

                        for _ in range(3):
                            latent_in_prev = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                            
                            # Use the current timestep tensor for model input scaling.
                            latent_in_prev = scheduler.scale_model_input(latent_in_prev, t_tensor)
                            noise_pred_prev = self.unet(
                                latent_in_prev, t_tensor, add_inputs=add_inputs,
                                encoder_hidden_states=None,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                timestep_cond=timestep_cond, coords=coords, return_dict=False,
                                cond_num=cond_num, class_labels=class_labels, added_cond_kwargs=added_cond_kwargs,
                                key_rescale=kwargs.get("key_rescale", None)
                            )[0]
                            
                            if self.do_classifier_free_guidance:
                                n_u, n_t = noise_pred_prev.chunk(2)
                                noise_pred_prev = n_u + self.guidance_scale * (n_t - n_u)

                            eps_full = noise_pred_prev.unsqueeze(0) if noise_pred_prev.dim() == 4 else noise_pred_prev
                            x_s_region = latents[:, tgt_idx]
                            eps_region = eps_full[:, tgt_idx, :x_s_region.shape[2]]

                            # Match the current noise level when reconstructing x0.
                            if pred_type == "epsilon":
                                x0_region = (x_s_region - (1.0 - alpha_bar_t).sqrt() * eps_region) / alpha_bar_t.sqrt()
                            elif pred_type == "v_prediction":
                                x0_region = alpha_bar_t.sqrt() * x_s_region - (1.0 - alpha_bar_t).sqrt() * eps_region
                            else:
                                x0_region = eps_region

                            x_t_region = alpha_bar_t.sqrt() * x0_region + (1.0 - alpha_bar_t).sqrt() * torch.randn_like(x0_region)
                            latents_xt = latents.detach().clone()
                            latents_xt[:, tgt_idx] = x_t_region
                            
                            latent_in_t = torch.cat([latents_xt] * 2) if self.do_classifier_free_guidance else latents_xt
                            latent_in_t = scheduler.scale_model_input(latent_in_t, t_tensor)
                            noise_pred_t = self.unet(
                                latent_in_t, t_tensor, add_inputs=add_inputs,
                                encoder_hidden_states=None,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                timestep_cond=timestep_cond, coords=coords, return_dict=False,
                                cond_num=cond_num, class_labels=class_labels, added_cond_kwargs=added_cond_kwargs,
                                key_rescale=kwargs.get("key_rescale", None)
                            )[0]
                            
                            if self.do_classifier_free_guidance:
                                n_u, n_t = noise_pred_t.chunk(2)
                                noise_pred_t = n_u + self.guidance_scale * (n_t - n_u)

                            latents_step = scheduler.step(noise_pred_t, t_tensor, latents_xt, **extra_step_kwargs, return_dict=False)[0]
                            latents[:, tgt_idx] = latents_step[:, tgt_idx]
                            latents[:, :cond_num] = image_latents[:, :cond_num]
                latents[:, :cond_num] = image_latents[:, :cond_num]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i // getattr(scheduler, "order", 1), t, latents)

        if latents.ndim == 5:
            latents = einops.rearrange(latents, "b f c h w -> (b f) c h w")

        if not output_type == "latent":
            sub_size = 8
            slice_num = (latents.shape[0] + sub_size - 1) // sub_size
            image = []
            for i in range(slice_num):
                img_ = self.vae.decode(latents[i * sub_size:(i + 1) * sub_size] / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
                image.append(img_)
            image = torch.cat(image, dim=0)
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=[True] * image.shape[0])
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, None)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=None)

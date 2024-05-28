# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from loguru import logger
from dataclasses import dataclass, field
from functools import partial
from copy import deepcopy
import torch
import transformers
from models.experimental.functional_vit.reference import torch_functional_vit
from ttnn.model_preprocessing import preprocess_model_parameters


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class BaseModelOutputWithPastAndCrossAttentions:
    def __init__(
        self,
        last_hidden_state: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None,
        attentions: Optional[Tuple[torch.FloatTensor, ...]] = None,
        cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None,
    ):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.cross_attentions = cross_attentions

    def __getitem__(self, item):
        return getattr(self, str(item))


class GenerateDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class AttentionMaskConverter:
    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.Tensor, attention_mask: torch.Tensor, unmasked_value: Union[bool, float]
    ):
        # fmt: off
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `attention_mask` is
        ```
        [[0, 0, 1],
         [1, 1, 1],
         [0, 1, 1]]
        ```
        and `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
        # fmt: on

        # Get the index of the first non-zero value for every sample in the batch.
        # In the above example, indices = [[2], [0], [1]]]
        tmp = torch.arange(attention_mask.shape[1], 0, -1)
        indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)

        # Find the batch indexes that have unattended tokens on the leftmost side (e.g. [0, 0, 1, 1, 1]), for which the first rows of the
        # expanded mask will be completely unattended.
        left_masked_rows = torch.where(indices > 0)[0]

        if left_masked_rows.shape[0] == 0:
            return expanded_mask
        indices = indices[left_masked_rows]

        max_len = torch.max(indices)
        range_tensor = torch.arange(max_len).unsqueeze(0)
        range_tensor = range_tensor.repeat(indices.size(0), 1)

        # Avoid unmasking tokens at relevant target positions (on the row axis), by rather unmasking possibly several times the first row that should always be unmasked as we filtered out the batch above.
        range_tensor[range_tensor >= indices] = 0

        # TODO: we may drop support for 3D attention mask as the refactor from Patrick maybe dropped this case
        if expanded_mask.dim() == 4:
            num_masks = expanded_mask.shape[1]
            if num_masks == 1:
                # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
                mask_slice = (left_masked_rows[:, None], 0, range_tensor)
            else:
                # Broadcast [left_masked_rows, 1, 1], [1, num_masks, 1], [left_masked_rows, 1, max_len]
                mask_slice = (
                    left_masked_rows[:, None, None],
                    torch.arange(num_masks)[None, :, None],
                    range_tensor[:, None, :],
                )
        else:
            # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
            mask_slice = (left_masked_rows[:, None], range_tensor)

        expanded_mask[mask_slice] = unmasked_value

        return expanded_mask


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


class LogitsProcessorList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


@dataclass
class GreedySearchDecoderOnlyOutput:
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


@dataclass
class GreedySearchEncoderDecoderOutput:
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None


GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
GenerateOutput = GreedySearchOutput
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]


def _prepare_model_inputs(
    inputs: Optional[torch.Tensor] = None,
    bos_token_id: Optional[int] = None,
    model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
    """
    This function extracts the model-specific `inputs` for generation.
    """
    # 1. retrieve all kwargs that are non-None or non-model input related.
    # some encoder-decoder models have different names for model and encoder
    input_name = "pixel_values"

    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

    return inputs, input_name, model_kwargs


def _prepare_attention_mask_for_generation(
    self,
    inputs: torch.Tensor,
    pad_token_id: Optional[int],
    eos_token_id: Optional[Union[int, List[int]]],
) -> torch.LongTensor:
    is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
    is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

    # Check if input is input_ids and padded -> only then is attention_mask defined
    if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
        return inputs.ne(pad_token_id).long()
    else:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)


def _prepare_encoder_decoder_kwargs_for_generation(
    encoder_name, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
):
    # 1. get encoder
    encoder = transformers.ViTModel.from_pretrained(encoder_name)
    encoder = encoder.to(torch.bfloat16)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: encoder,
        convert_to_ttnn=lambda *_: False,
        custom_preprocessor=torch_functional_vit.custom_preprocessor,
    )
    model_state_dict = encoder.state_dict()
    torch_cls_token = torch.nn.Parameter(model_state_dict["embeddings.cls_token"])
    torch_position_embeddings = torch.nn.Parameter(model_state_dict["embeddings.position_embeddings"])

    # 2. Prepare encoder args and encoder kwargs from model kwargs.
    irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
    encoder_kwargs = {
        argument: value
        for argument, value in model_kwargs.items()
        if not any(argument.startswith(p) for p in irrelevant_prefix)
    }
    encoder_signature = set(inspect.signature(encoder.forward).parameters)
    # print("\n encoder_signature:", encoder_signature)
    encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
    if not encoder_accepts_wildcard:
        encoder_kwargs = {
            argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
        }

    # 3. make sure that encoder returns `ModelOutput`
    model_input_name = "pixel_values"
    model_kwargs["return_dict"] = False
    model_kwargs["output_attentions"] = False
    model_kwargs["output_hidden_states"] = False

    encoder_op_list = torch_functional_vit.vit(
        encoder.config,
        inputs_tensor,
        torch_position_embeddings,
        torch_cls_token,
        attention_mask=None,
        parameters=parameters,
    )

    model_kwargs["encoder_outputs"] = encoder_op_list[0]
    # print(len(encoder_op_list), type(encoder_op_list), encoder_op_list[0].shape, encoder_op_list[1].shape)
    # exit()
    return model_kwargs


def _get_decoder_start_token_id(
    decoder_start_token_id: Union[int, List[int]] = None,
    bos_token_id: int = None,
    generation_config=None,
) -> int:
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else generation_config.decoder_start_token_id
    )
    bos_token_id = bos_token_id if bos_token_id is not None else generation_config.bos_token_id

    if decoder_start_token_id is not None:
        return decoder_start_token_id
    elif bos_token_id is not None:
        return bos_token_id
    raise ValueError("`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation.")


class LogitsProcessorList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


from abc import ABC


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )
        return is_done


class MaxTimeCriteria(StoppingCriteria):
    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return time.time() - self.initial_timestamp > self.max_time


class StoppingCriteriaList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None


def _merge_criteria_processor_list(
    default_list: Union[LogitsProcessorList, StoppingCriteriaList],
    custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
) -> Union[LogitsProcessorList, StoppingCriteriaList]:
    if len(custom_list) == 0:
        return default_list
    for default in default_list:
        for custom in custom_list:
            if type(custom) is type(default):
                object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                raise ValueError(
                    f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                    f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                    " created by passing the corresponding arguments to generate or by the model's config default"
                    f" values. If you just want to change the default values of {object_type} consider passing"
                    f" them as arguments to `.generate()` instead of using a custom {object_type}."
                )
    default_list.extend(custom_list)
    return default_list


class GenerateEncoderDecoderOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None


def _prepare_decoder_input_ids_for_generation(
    config,
    batch_size: int,
    model_input_name: str,
    model_kwargs: Dict[str, torch.Tensor],
    decoder_start_token_id: Union[int, List[int]] = None,
    bos_token_id: int = None,
    device: torch.device = None,
) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
    """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
    # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
    # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
    if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
        decoder_input_ids = model_kwargs.pop("decoder_input_ids")
    elif "input_ids" in model_kwargs and model_input_name != "input_ids":
        decoder_input_ids = model_kwargs.pop("input_ids")
    else:
        decoder_input_ids = None

    # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
    decoder_start_token_id = _get_decoder_start_token_id(decoder_start_token_id, bos_token_id)

    if isinstance(decoder_start_token_id, list):
        if len(decoder_start_token_id) != batch_size:
            raise ValueError(
                f"`decoder_start_token_id` expcted to have length {batch_size} but got {len(decoder_start_token_id)}"
            )
        decoder_input_ids_start = torch.tensor(decoder_start_token_id, dtype=torch.long)
        decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
    else:
        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long) * decoder_start_token_id

    # no user input -> use decoder_start_token_id as decoder_input_ids
    if decoder_input_ids is None:
        decoder_input_ids = decoder_input_ids_start
    # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
    elif config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
        pass
    elif config.model_type in ["whisper"]:
        pass
    # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
    # decoder_attention_mask if provided)
    elif (
        isinstance(decoder_start_token_id, int) and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
    ) or (
        isinstance(decoder_start_token_id, torch.Tensor)
        and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
    ):
        decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            decoder_attention_mask = torch.cat(
                (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                dim=-1,
            )
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask

    return decoder_input_ids, model_kwargs


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria


def decoder_prepare_inputs(input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs):
    # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    if past_key_values:
        past_length = past_key_values[0][0].shape[2]

        # Some generation methods already pass only the last input ID
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            # Default to old behavior: keep only final ID
            remove_prefix_length = input_ids.shape[1] - 1

        input_ids = input_ids[:, remove_prefix_length:]
    # first step, decoder_cached_states are empty
    return {
        "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
    }


def prepare_inputs_for_generation(
    input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
):
    decoder_inputs = decoder_prepare_inputs(input_ids, past_key_values=past_key_values)
    decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
    input_dict = {
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "decoder_input_ids": decoder_inputs["input_ids"],
        "encoder_outputs": encoder_outputs,
        "past_key_values": decoder_inputs["past_key_values"],
        "use_cache": use_cache,
    }
    print("\n dug ...", decoder_attention_mask)
    return input_dict


def _update_model_kwargs_for_generation(
    outputs,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
) -> Dict[str, Any]:
    # update past_key_values
    model_kwargs["past_key_values"] = None
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    # update token_type_ids with last value
    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    return model_kwargs


def greedy_search(
    config,
    input_ids: torch.LongTensor,
    generation_config,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    output_logits: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else generation_config.output_scores
    output_attentions = output_attentions if output_attentions is not None else generation_config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    while True:
        # prepare model inputs
        model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)

        decoder_name = "microsoft/trocr-base-handwritten"
        encoder_name = "google/vit-base-patch16-224"
        from models.experimental.functional_trocr.reference.functional_vision_en_dec import encoder_decoder

        outputs = encoder_decoder(
            encoder_name,
            decoder_name,
            **model_inputs,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs[:, -1, :]

        # pre-process distribution
        next_tokens_scores = next_token_logits  # logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if config.is_encoder_decoder else (outputs.attentions,)
                )
                if config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,) if config.is_encoder_decoder else (outputs.hidden_states,)
                )

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = _update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids


def _get_stopping_criteria(
    config, generation_config, stopping_criteria: Optional[StoppingCriteriaList]
) -> StoppingCriteriaList:
    criteria = StoppingCriteriaList()
    if generation_config.max_length is not None:
        max_position_embeddings = getattr(config, "max_position_embeddings", None)
        criteria.append(
            MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings,
            )
        )
    if generation_config.max_time is not None:
        criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
    criteria = _merge_criteria_processor_list(criteria, stopping_criteria)
    return criteria


def generate(
    config,
    generation_config,
    encoder_name,
    decoder_name,
    inputs: Optional[torch.Tensor] = None,
    # generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    streamer: Optional["BaseStreamer"] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    model_kwargs = generation_config.update(**kwargs)

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

        # 3.Define model inputs
    inputs_tensor, model_input_name, model_kwargs = _prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states

    if not config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = False  # "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs
    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = _prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = _prepare_encoder_decoder_kwargs_for_generation(
            encoder_name, inputs_tensor, model_kwargs, model_input_name
        )

    if config.is_encoder_decoder:
        input_ids, model_kwargs = _prepare_decoder_input_ids_for_generation(
            config=config,
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
        )

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length

    # 8. prepare distribution pre_processing samplers, logits_processor is empty list

    # 9. prepare stopping criteria
    prepared_stopping_criteria = _get_stopping_criteria(
        config=config, generation_config=generation_config, stopping_criteria=stopping_criteria
    )

    return greedy_search(
        config,
        input_ids,
        generation_config,
        logits_processor=[],
        stopping_criteria=prepared_stopping_criteria,
        pad_token_id=generation_config.pad_token_id,
        eos_token_id=generation_config.eos_token_id,
        output_scores=generation_config.output_scores,
        output_logits=generation_config.output_logits,
        return_dict_in_generate=generation_config.return_dict_in_generate,
        synced_gpus=synced_gpus,
        streamer=streamer,
        **model_kwargs,
    )

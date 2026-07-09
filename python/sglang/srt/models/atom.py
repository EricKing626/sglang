# Copyright 2026 SGLang Team
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
# ==============================================================================
"""Wrapper around `atom` models.

Aligned with the ROCm/ATOM plugin API (atom.plugin.sglang): the ATOM model
object returned by ``prepare_model_for_sglang`` is a full SGLang-compatible
wrapper that already owns the ``LogitsProcessor`` / lm_head contract and returns
a ``LogitsProcessorOutput`` directly. This adapter therefore only forwards the
SGLang forward-call and delegates weight loading; it must NOT run a second
LogitsProcessor of its own.
"""
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class ATOMForCausalLM(nn.Module):

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Using Atom backend.")

        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        import atom

        # Current ROCm/ATOM exports `prepare_model_for_sglang(config)`.
        self.model = atom.prepare_model_for_sglang(config)
        if self.model is None:
            model_arch = getattr(config, "architectures", ["unknown"])[0]
            raise ValueError(f"This model {model_arch} is not supported by atom")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
    ) -> LogitsProcessorOutput:
        # The ATOM wrapper already applies the LogitsProcessor internally and
        # returns a LogitsProcessorOutput on the last pipeline rank.
        return self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            get_embedding=get_embedding,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return self.model.load_weights(weights)


EntryClass = [ATOMForCausalLM]

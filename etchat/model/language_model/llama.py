# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM, LlamaModel

from etchat.model.etchat_arch import ETChatMetaForCausalLM, ETChatMetaModel


class ETChatLlamaConfig(LlamaConfig):
    model_type = 'etchat_llama'


class ETChatLlamaModel(ETChatMetaModel, LlamaModel):
    config_class = ETChatLlamaConfig


class ETChatLlamaForCausalLM(ETChatMetaForCausalLM, LlamaForCausalLM):
    config_class = ETChatLlamaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = ETChatLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


AutoConfig.register('etchat_llama', ETChatLlamaConfig)
AutoModelForCausalLM.register(ETChatLlamaConfig, ETChatLlamaForCausalLM)

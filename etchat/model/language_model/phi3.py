# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Phi3Config, Phi3ForCausalLM, Phi3Model

from etchat.model.etchat_arch import ETChatMetaForCausalLM, ETChatMetaModel


class ETChatPhi3Config(Phi3Config):
    model_type = 'etchat_phi3'


class ETChatPhi3Model(ETChatMetaModel, Phi3Model):
    config_class = ETChatPhi3Config


class ETChatPhi3ForCausalLM(ETChatMetaForCausalLM, Phi3ForCausalLM):
    config_class = ETChatPhi3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = ETChatPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()


AutoConfig.register('etchat_phi3', ETChatPhi3Config)
AutoModelForCausalLM.register(ETChatPhi3Config, ETChatPhi3ForCausalLM)

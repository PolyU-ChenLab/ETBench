# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

from .language_model.gemma2 import ETChatGemma2Config, ETChatGemma2ForCausalLM
from .language_model.llama import ETChatLlamaConfig, ETChatLlamaForCausalLM
from .language_model.phi3 import ETChatPhi3Config, ETChatPhi3ForCausalLM
from .language_model.qwen2 import ETChatQwen2Config, ETChatQwen2ForCausalLM

ETCHAT_MODELS = {
    'llama': (ETChatLlamaConfig, ETChatLlamaForCausalLM),
    'phi3': (ETChatPhi3Config, ETChatPhi3ForCausalLM),
    'gemma2': (ETChatGemma2Config, ETChatGemma2ForCausalLM),
    'qwen2': (ETChatQwen2Config, ETChatQwen2ForCausalLM)
}

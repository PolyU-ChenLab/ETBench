# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import torch
from transformers import LogitsProcessor, StoppingCriteria, TextIteratorStreamer

from .tokenization import detokenize


class MatchStreamer(TextIteratorStreamer):

    def __init__(self, model, *args, **kwargs):
        super(MatchStreamer, self).__init__(*args, **kwargs)
        self.model = model

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError('MatchStreamer only supports batch size 1')
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(value.tolist())
        text = detokenize(self.token_cache, self.model, self.tokenizer, **self.decode_kwargs)

        if text.endswith('\n'):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len:text.rfind(' ') + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):

    def __init__(self, penalty, match_token_id):
        assert isinstance(penalty, (int, float)) and penalty > 0
        self.penalty = penalty
        self.match_token_id = match_token_id

    def __call__(self, input_ids, scores):
        assert input_ids.shape[0] == 1

        _input_ids = input_ids[0][(input_ids[0] != -200) * (input_ids[0] != self.match_token_id)].unsqueeze(0)
        if _input_ids.shape[1] == 0:
            return scores

        score = torch.gather(scores, 1, _input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores.scatter_(1, _input_ids, score)
        return scores


class KeywordsStoppingCriteria(StoppingCriteria):

    def __init__(self, tokenizer, keywords):
        if not isinstance(keywords, (list, tuple)):
            keywords = [keywords]

        self.keyword_ids = []
        for keyword in keywords:
            kid = tokenizer(keyword, return_tensors='pt').input_ids[0]
            if kid[0] == tokenizer.bos_token_id:
                kid = kid[1:]
            self.keyword_ids.append(kid)

    def __call__(self, input_ids, *args, **kwargs):
        assert input_ids.size(0) == 1

        for kid in self.keyword_ids:
            if (input_ids[0][-kid.size(0):] == kid.to(input_ids.device)).all():
                return True

        return False

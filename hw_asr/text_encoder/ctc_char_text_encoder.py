from typing import List, NamedTuple

import torch
from collections import defaultdict

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        last_char = self.EMPTY_TOK
        res = []
        for ind in inds:
            cur_char = self.ind2char[ind]

            if cur_char == self.EMPTY_TOK:
                last_char = self.EMPTY_TOK
                continue

            if cur_char != last_char:
                res.append(cur_char)
                last_char = cur_char
    
        return ''.join(res)


    def _extend_and_merge(self, current_state, dist):
        new_state = defaultdict(float)

        for (prefix, last_char), prefix_p in current_state.items():
            for next_char_index, next_char_p in enumerate(dist):
                next_char = self.ind2char[next_char_index]

                if next_char == last_char or next_char == self.EMPTY_TOK:
                    new_pref = prefix
                else:
                    new_pref = prefix + next_char

                last_char = next_char
                new_state[(new_pref, last_char)] += prefix_p * next_char_p.item()
        
        return new_state


    def _truncate(self, current_state, beam_size):
        current_state = sorted(current_state.items(), key=lambda x: -x[1])
        return dict(current_state[:beam_size])


    def ctc_beam_search(self, probs: torch.tensor, probs_len: torch.tensor, beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        probs = torch.softmax(probs[:probs_len], dim=1)

        hypos: List[Hypothesis] = []
        state = {('', self.EMPTY_TOK): 1.0}
        for dist in probs:
            state = self._extend_and_merge(state, dist)
            state = self._truncate(state, beam_size)
        hypos = [Hypothesis(''.join(text), p) for (text, _), p in state.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    

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


    def ctc_beam_search(self, probs, probs_length, beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        :param probs: probabilities from model of shape [L, H]
        :param beam_size: size of beam to use in decoding

        Note: unpadding of probs should be done before passing to this function
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
    
        beam = defaultdict(float)

        probs = torch.softmax(probs[:probs_length], dim=1)

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._cut_beam(beam, beam_size)

        final_beam = defaultdict(float)
        for (sentence, last_char), v in beam.items():
            final_sentence = (sentence + last_char).strip().replace(self.EMPTY_TOK, "")
            final_beam[final_sentence] += v
            
        sorted_beam = sorted(final_beam.items(), key=lambda x: -x[1])
        result = [Hypothesis(sentence, v) \
                for sentence, v in sorted_beam]
        return result

    def _extend_beam(self, beam, prob):
        if len(beam) == 0:
            for i in range(len(prob)):
                last_char = self.ind2char[i]
                beam[('', last_char)] += prob[i]
            return beam

        new_beam = defaultdict(float)
        
        for (sentence, last_char), v in beam.items():
            for i in range(len(prob)):
                if self.ind2char[i] == last_char:
                    new_beam[(sentence, last_char)] += v * prob[i]
                else:
                    new_last_char = self.ind2char[i]
                    new_sentence = (sentence + last_char).replace(self.EMPTY_TOK, '')\
                                    .replace("'", "").replace("|", "")
                    new_beam[(new_sentence, new_last_char)] += v * prob[i]

        return new_beam

    def _cut_beam(self, beam, beam_size):
        return dict(sorted(beam.items(), key=lambda x: -x[1])[:beam_size])
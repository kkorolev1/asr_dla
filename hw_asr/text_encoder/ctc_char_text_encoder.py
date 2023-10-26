from typing import List, NamedTuple

import torch
from collections import defaultdict

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder
import multiprocessing


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, with_lm=False, kenlm_path=None, vocab_path=None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        self.with_lm = with_lm

        if self.with_lm:
            assert kenlm_path is not None, "Empty KENLM path"
            assert vocab_path is not None, "Empty vocab path"
            with open(vocab_path) as f:
                unigrams = [w.strip() for w in f.readlines()]
            self.decoder = build_ctcdecoder([""] + [w.upper() for w in self.alphabet], kenlm_model_path=kenlm_path, unigrams=unigrams)

    def ctc_decode(self, inds: List[int]) -> str:
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

    def _extend_beam(self, beam, prob):
        if len(beam) == 0:
            for i in range(len(prob)):
                last_char = self.ind2char[i]
                beam[('', last_char)] += prob[i]
            return beam

        new_beam = defaultdict(float)
        
        for (prefix, last_char), prefix_proba in beam.items():
            for i in range(len(prob)):
                cur_char = self.ind2char[i]
                if cur_char == last_char:
                    new_beam[(prefix, last_char)] += prefix_proba * prob[i]
                else:
                    new_last_char = cur_char
                    new_prefix = (prefix + last_char).replace(self.EMPTY_TOK, '')
                    new_beam[(new_prefix, new_last_char)] += prefix_proba * prob[i]
        return new_beam

    def _truncate_beam(self, beam, beam_size):
        return dict(sorted(beam.items(), key=lambda x: -x[1])[:beam_size])

    def ctc_beam_search(self, probs, probs_length, beam_size: int = 100) -> List[Hypothesis]:
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
    
        beam = defaultdict(float)
        probs = probs[:probs_length]

        for prob in probs:
            beam = self._extend_beam(beam, prob)
            beam = self._truncate_beam(beam, beam_size)

        final_beam = defaultdict(float)
        for (sentence, last_char), p in beam.items():
            final_sentence = (sentence + last_char).strip().replace(self.EMPTY_TOK, "")
            final_beam[final_sentence] += p
            
        sorted_beam = sorted(final_beam.items(), key=lambda x: -x[1])
        return [Hypothesis(text, p.item()) for text, v in sorted_beam]
    
    def ctc_lm_beam_search(self, probs, probs_length, beam_size: int = 100) -> List[Hypothesis]:
        assert probs.shape[-1] == len(self.ind2char)

        if len(probs.shape) == 2:
            probs = probs.unsqueeze(0)

        probs = [probs[i][:probs_length[i]].numpy() for i in range(probs_length.shape[0])]

        with multiprocessing.get_context("fork").Pool() as pool:
            texts = self.decoder.decode_batch(pool, probs, beam_width=beam_size)
        
        return [w.lower().replace("'", "").replace("|", "").replace("??", "").strip() for w in texts]

import unittest

import torch
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        # TODO: (optional) write tests for beam search
        text_encoder = CTCCharTextEncoder()

        true_text = "i wish i started doing this hw earlier"
        inds = torch.tensor([text_encoder.char2ind[c] for c in true_text])
        
        probs = torch.zeros((len(true_text), len(text_encoder)))
        probs[torch.arange(probs.shape[0]), inds] = 1
        
        hypos = text_encoder.ctc_beam_search(probs, probs.shape[0], beam_size=3)
        print(hypos)

        
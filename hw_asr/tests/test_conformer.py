import unittest

from tqdm import tqdm


from hw_asr.tests.utils import clear_log_folder_after_use
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.model.conformer import Conformer

class TestConformer(unittest.TestCase):
    def test_dataloaders(self):
        _TOTAL_ITERATIONS = 10
        config_parser = ConfigParser.get_test_configs()
        
        n_class = 10
        model = Conformer(n_feats=128, n_class=n_class, encoder_layers=4, attention_heads=8, decoder_layers=3)

        with clear_log_folder_after_use(config_parser):
            dataloaders = get_dataloaders(config_parser, config_parser.get_text_encoder())
            for part in ["train", "val"]:
                dl = dataloaders[part]
                for i, batch in tqdm(enumerate(iter(dl)), total=_TOTAL_ITERATIONS,
                                     desc=f"Iterating over {part}"):
                    logits = model(**batch)
                    input_shape = batch['spectrogram'].shape
                    output_shape = logits.shape
                    self.assertEqual(output_shape[-1], n_class)
                    self.assertEqual((input_shape[-1] - 3) // 4, output_shape[1])
                    break

    def test_length(self):
        _TOTAL_ITERATIONS = 1000
        config_parser = ConfigParser.get_test_configs()

        with clear_log_folder_after_use(config_parser):
            dataloaders = get_dataloaders(config_parser, config_parser.get_text_encoder())
            for part in ["train", "val"]:
                dl = dataloaders[part]
                max_len = 0
                for i, batch in tqdm(enumerate(iter(dl)), total=_TOTAL_ITERATIONS,
                                     desc=f"Iterating over {part}"):
                    len = batch['spectrogram'].shape[-1]
                    max_len = max(max_len, len)
                print(f"{part} max len = {max_len}")

                    
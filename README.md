# ASR HW 1

Automatic Speech Recognition task using Conformer model trained on Librispeech.

## Installation guide

```shell
pip install -r ./requirements.txt
```

Download 3-gram.arpa and vocab from https://www.openslr.org/11/. To use them change kenlm_path and vocab_path in config.json. You can download model checkpoint from [this link](https://drive.google.com/file/d/1DBNBP8ap7NvWtEQ8g4GrnEXstz-UG3-m/view?usp=sharing).

## Training
```shell
python train.py -c CONFIG
```
Check hw_asr for config examples

## Testing
One can choose which metrics to evaluate in config file.
```shell
python test.py -c CONFIG -r CHECKPOINT
```
Eventually, you will have a file with predictions, metrics are outputted to the console.

# ASR HW 1

Automatic Speech Recognition task using Conformel model trained on Librispeech.

## Installation guide

```shell
pip install -r ./requirements.txt
```

Download 3-gram.arpa and vocab from https://www.openslr.org/11/. To use them change kenlm_path and vocab_path in config.json.

## Training
```shell
python train.py -c CONFIG
```
Check hw_asr for config examples

## Testing
```shell
python test.py -c CONFIG -r CHECKPOINT
```

# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
  <a href="#report">Report</a>
</p>


## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=3.11

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/3.11.13/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the the following:
1. Firstly, download all necessary resources for training. You can do it with script `load_resources.py`:
```bash
python3 load_resources.py --output output_dir
```
2. Then run training:
```bash
python3 train.py data_dir=output_dir/resources/librispeech lm_guidance_dir=output_dir/resources/lm_guidance aug_dir=output_dir/resources/aug_data
```

2.5 To launch training with pretrained HuggingFace BPE Tokenizer, run:
```bash
python3 train.py data_dir=output_dir/resources/librispeech lm_guidance_dir=output_dir/resources/lm_guidance aug_dir=output_dir/resources/aug_data tokenizer_config.save_path=your_tokenizer.json tokenizer_config.use_tokenizer=True
```

To train tokenizer before training run:
```bash
python3 train.py data_dir=output_dir/resources/librispeech lm_guidance_dir=output_dir/resources/lm_guidance aug_dir=output_dir/resources/aug_data tokenizer_config.save_path=save_dir/tokenizer.json tokenizer_config.use_tokenizer=True
```

Default config used for training is situated is `src/configs/baseline.yaml`.

To run inference for saving predictions:

```bash
python3 inference.py data_dir=your_data inferencer.save_path=your_save_dir inferencer.from_pretrained=output_dir/resources/ckpt/model_best.pth lm_guidance_dir=output_dir/resources/lm_guidance
```
To run model evaluation on predicted and GT transcriptions:
```bash
python3 calc_metrics.py --predictions your_save_dir --ground_truth your_data/transcriptions
```
Examples of inference and evaluation scripts usage and supported dataset structure can be found in `demo.ipynb` notebook.
## Report
Report can be found [here](https://api.wandb.ai/links/ayazbebrovich-hse-fcs/waatwb97).

## Credits

1. This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).
2. ASR model is implemented based on the [original Deepspeech2 paper](https://arxiv.org/abs/1512.02595).
3. Augmentation dataset was downloaded from [OpenSLR](https://www.openslr.org/28/)
4. LM arpa was downloaded from [OpenSLR](https://www.openslr.org/11)

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

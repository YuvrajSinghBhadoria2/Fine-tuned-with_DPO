# Fine-tuned-with_DPO

This repository contains an implementation of Direct Preference Optimization (DPO) for fine-tuning language models. It is based on the original DPO algorithm and includes support for conservative DPO and IPO variants.

## About This Project

This project demonstrates how to fine-tune language models using Direct Preference Optimization, a method that trains language models directly from preference data without requiring a separate reward model.

## Key Features

- Implementation of Direct Preference Optimization (DPO)
- Support for Conservative DPO and IPO variants
- Configurable training pipelines for various HuggingFace models
- Flexible dataset handling for preference-based training

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Training Process

The DPO pipeline consists of two stages:

1. **Supervised Fine-tuning (SFT)**: Train the base model on your target dataset
2. **Preference Learning**: Fine-tune using preference data with DPO

### Example Usage

Train a Pythia 2.8B model on Anthropic-HH dataset:

```bash
# Stage 1: SFT
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=my_experiment gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false

# Stage 2: DPO
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=my_experiment gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.archive=/path/to/sft/checkpoint/policy.pt
```

## Customization

You can customize the training by modifying configurations in:
- `config/config.yaml`: Main configuration options
- `config/model/`: Model-specific configurations
- `config/loss/`: Loss function configurations

Add your own datasets by modifying `preference_datasets.py`.

## Acknowledgements

This implementation is based on the original DPO research:
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

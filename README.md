# Fine-tuned-with_DPO

This repository contains an implementation of Direct Preference Optimization (DPO) for fine-tuning language models. It is based on the original DPO algorithm and includes support for conservative DPO and IPO variants.

## About This Project

This project demonstrates how to fine-tune language models using Direct Preference Optimization, a method that trains language models directly from preference data without requiring a separate reward model. This approach simplifies the alignment process by eliminating the need for reward modeling.

## Key Features

- Implementation of Direct Preference Optimization (DPO)
- Support for Conservative DPO and IPO variants
- Configurable training pipelines for various HuggingFace models
- Flexible dataset handling for preference-based training
- Multi-GPU training support with FSDP

## Expected Outcomes

When using this DPO implementation, you can expect:

### Model Performance Improvements
- Better alignment of language models with human preferences
- Improved response quality and helpfulness
- Reduced harmful or undesirable outputs
- More consistent and reliable responses

### Training Efficiency
- Faster convergence compared to traditional reinforcement learning approaches
- Elimination of the reward modeling step, reducing overall training time
- Lower computational requirements than multi-stage alignment methods

### Practical Benefits
- Simplified training pipeline with fewer hyperparameters to tune
- Direct optimization of the policy based on preference data
- Better sample efficiency compared to RL-based methods

## Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- CUDA-compatible GPU (recommended for training)

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

## Configuration Options

### Model Configurations
Customize your training by modifying configurations in:
- `config/config.yaml`: Main configuration options
- `config/model/`: Model-specific configurations
- `config/loss/`: Loss function configurations

### Key Parameters
- `loss.beta`: Controls the strength of the DPO regularization (typically 0.1-0.5)
- `batch_size`: Total batch size across all GPUs
- `gradient_accumulation_steps`: Used to simulate larger batch sizes
- `trainer`: Select between BasicTrainer, FSDPTrainer, or TensorParallelTrainer

## Adding Custom Datasets

Add your own datasets by modifying `preference_datasets.py`. Follow the existing patterns for:
- `get_hh()`: Anthropic Helpfulness and Harmlessness dataset
- `get_shp()`: Stanford Human Preferences dataset
- `get_se()`: StackExchange dataset

Each dataset function should return a dictionary mapping prompts to responses, preference pairs, and SFT targets.

## Multi-GPU Training

For faster training on multiple GPUs:
1. Use `trainer=FSDPTrainer` for FSDP-based training
2. Enable mixed precision with `model.fsdp_policy_mp=bfloat16`
3. Consider activation checkpointing with `activation_checkpointing=true` for memory-constrained setups

## Troubleshooting

Common issues and solutions:
- If encountering memory issues, reduce batch size or enable gradient accumulation
- For sampling speed issues with FSDP, disable sampling with `sample_during_eval=false`
- Ensure proper file permissions when running distributed training

## Acknowledgements

This implementation is based on the original DPO research:
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

Additional variants supported:
- [Conservative DPO](https://ericmitchell.ai/cdpo.pdf)
- [IPO: Implicit Preference Optimization](https://arxiv.org/pdf/2310.12036.pdf)

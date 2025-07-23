# BERT Masked Language Model Training Example

This repository provides a minimal, clean implementation of BERT (Bidirectional Encoder Representations from Transformers) training for the Masked Language Modeling (MLM) task.

## Table of Contents
- [What is BERT?](#what-is-bert)
- [How BERT Works](#how-bert-works)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Dataset Requirements](#dataset-requirements)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)

## What is BERT?

BERT (Bidirectional Encoder Representations from Transformers) is a revolutionary pre-trained language model developed by Google in 2018. Unlike traditional language models that read text sequentially (left-to-right or right-to-left), BERT reads text bidirectionally, allowing it to understand context from both directions simultaneously.

### Key Features:
- **Bidirectional Context**: BERT considers both left and right context when understanding words
- **Pre-training and Fine-tuning**: BERT is first pre-trained on large text corpora, then fine-tuned for specific tasks
- **Transformer Architecture**: Built on the powerful attention mechanism that revolutionized NLP
- **Transfer Learning**: Pre-trained BERT models can be adapted to various downstream tasks

## How BERT Works

### 1. Masked Language Modeling (MLM)
BERT's primary training objective is to predict randomly masked tokens in a sentence. For example:
- Input: "The [MASK] brown fox jumps over the lazy dog"
- Task: Predict that [MASK] = "quick"

The masking strategy:
- 15% of tokens are selected for masking
- Of these:
  - 80% are replaced with [MASK]
  - 10% are replaced with random tokens
  - 10% are left unchanged

### 2. Architecture Components

#### Tokenization
BERT uses WordPiece tokenization to handle vocabulary:
- Breaks words into subword units
- Handles out-of-vocabulary words effectively
- Special tokens: [CLS] (classification), [SEP] (separator), [MASK] (masking), [PAD] (padding)

#### Embeddings
BERT combines three types of embeddings:
1. **Token Embeddings**: Represent individual tokens
2. **Position Embeddings**: Encode position information (up to 512 tokens)
3. **Segment Embeddings**: Distinguish between different text segments

#### Transformer Layers
- **Self-Attention**: Allows each token to attend to all other tokens
- **Multi-Head Attention**: Captures different types of relationships
- **Feed-Forward Networks**: Process attention outputs
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Enable deep networks

## Repository Structure

```
masked/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── model.py                 # BERT model implementation
├── data_loader.py           # Dataset and data loading utilities
├── train.py                 # Training script
├── inference.py             # Inference and testing script
├── datasets/                # Sample datasets (created during training)
│   └── sample_data.txt      # Example training texts
└── models/                  # Saved model checkpoints
    ├── best_model/          # Best performing model
    ├── final_model/         # Final model after all epochs
    └── checkpoint_epoch_*   # Epoch checkpoints
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.0+
- CUDA (optional, for GPU acceleration)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

1. **Clone the repository**:
```bash
git clone <repository-url>
cd masked
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train the model**:
```bash
python train.py
```

4. **Test the trained model**:
```bash
python inference.py --model_path models/best_model
```

## Dataset Requirements

### Format
- Plain text files with one or more sentences per line
- UTF-8 encoding
- No special formatting required

### Size Recommendations
- Minimum: 10,000 sentences for basic demonstration
- Recommended: 100,000+ sentences for meaningful results
- Optimal: Millions of sentences for production models

### Language Support
- Default configuration uses English (bert-base-uncased)
- Multilingual models available (bert-base-multilingual-cased)
- Language-specific models can be specified via --model_name

### Creating Custom Datasets
Place your text data in a `.txt` file and modify the `load_sample_data()` function in `data_loader.py`:

```python
def load_custom_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    return [text.strip() for text in texts if text.strip()]
```

## Training

### Basic Training
```bash
python train.py
```

### Advanced Training Options
```bash
python train.py \
    --model_name bert-base-uncased \
    --batch_size 16 \
    --epochs 5 \
    --learning_rate 5e-5 \
    --max_length 128 \
    --mlm_probability 0.15 \
    --save_dir models
```

### Training Parameters
- `--model_name`: Pre-trained model to use (default: bert-base-uncased)
- `--batch_size`: Training batch size (default: 16)
- `--epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--max_length`: Maximum sequence length (default: 128)
- `--mlm_probability`: Probability of masking tokens (default: 0.15)
- `--warmup_steps`: Learning rate warmup steps (default: 100)
- `--save_dir`: Directory to save models (default: models)

### Training Process
1. Loads pre-trained BERT model
2. Prepares training data with masked tokens
3. Trains model to predict masked tokens
4. Evaluates on validation set after each epoch
5. Saves best model based on validation loss

## Inference

### Interactive Testing
```bash
python inference.py --model_path models/best_model
```

### Batch Prediction
```python
from inference import BERTInference

# Load model
bert = BERTInference('models/best_model')

# Predict masked tokens
text = "The [MASK] is shining brightly today."
predictions = bert.predict_masked(text)
print(predictions)
```

### Fill Multiple Masks
```python
text = "BERT is a [MASK] language [MASK] model."
filled_text = bert.fill_masks(text, top_k=3)
print(filled_text)
```

## Model Architecture

### BERT Base
- Layers: 12 transformer blocks
- Hidden Size: 768
- Attention Heads: 12
- Parameters: ~110M
- Max Sequence Length: 512

### BERT Large
- Layers: 24 transformer blocks
- Hidden Size: 1024
- Attention Heads: 16
- Parameters: ~340M
- Max Sequence Length: 512

### Memory Requirements
- BERT Base: ~4GB GPU memory for batch size 16
- BERT Large: ~16GB GPU memory for batch size 16

## Tips for Best Results

1. **Data Quality**: Use clean, diverse text data
2. **Batch Size**: Larger batches generally improve training stability
3. **Learning Rate**: Start with 5e-5 and adjust based on results
4. **Training Time**: More epochs generally improve performance (with diminishing returns)
5. **GPU Usage**: Training is significantly faster with CUDA-enabled GPUs

## Common Issues and Solutions

### Out of Memory
- Reduce batch_size
- Reduce max_length
- Use gradient accumulation
- Use a smaller model (e.g., distilbert-base-uncased)

### Poor Performance
- Increase training data
- Train for more epochs
- Adjust learning rate
- Check data quality

### Slow Training
- Use GPU acceleration
- Reduce max_length if possible
- Use mixed precision training

## References

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
3. [The Illustrated BERT](http://jalammar.github.io/illustrated-bert/)

## License

This is an educational example implementation. Please refer to the original BERT paper and Hugging Face licenses for model usage rights.
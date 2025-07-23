import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import random
from typing import List, Dict, Tuple


class MaskedLanguageModelingDataset(Dataset):
    """
    Dataset for Masked Language Modeling task.
    Randomly masks tokens in the input text for BERT to predict.
    """
    
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, max_length: int = 128, mlm_probability: float = 0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels (copy of input_ids)
        labels = input_ids.clone()
        
        # Create mask for MLM
        masked_input_ids, labels = self._mask_tokens(input_ids, labels)
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _mask_tokens(self, input_ids: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        masked_input_ids = input_ids.clone()
        
        # Create random mask
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = torch.tensor(
            self.tokenizer.get_special_tokens_mask(input_ids.tolist(), already_has_special_tokens=True),
            dtype=torch.bool
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Don't mask padding tokens
        padding_mask = input_ids == self.tokenizer.pad_token_id
        probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Create mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # For non-masked tokens, set labels to -100 (ignored in loss calculation)
        labels[~masked_indices] = -100
        
        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        masked_input_ids[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        masked_input_ids[indices_random] = random_words[indices_random]
        
        # The rest 10% of the time, keep masked input tokens unchanged
        
        return masked_input_ids, labels


def create_dataloader(
    texts: List[str],
    tokenizer: BertTokenizer,
    batch_size: int = 16,
    max_length: int = 128,
    mlm_probability: float = 0.15,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a DataLoader for masked language modeling.
    
    Args:
        texts: List of text strings
        tokenizer: BERT tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        mlm_probability: Probability of masking each token
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    dataset = MaskedLanguageModelingDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        mlm_probability=mlm_probability
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )


def load_sample_data() -> List[str]:
    """
    Load sample training data for demonstration.
    In a real scenario, this would load from files or datasets.
    """
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "BERT stands for Bidirectional Encoder Representations from Transformers.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models have revolutionized computer vision and NLP.",
        "Transformers architecture has become the foundation of modern NLP.",
        "Attention mechanism allows models to focus on relevant parts of input.",
        "Pre-training on large corpora helps models learn general language patterns.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Masked language modeling is a self-supervised training objective.",
        "The model learns to predict masked tokens based on context.",
        "Bidirectional context improves understanding of word meanings.",
        "BERT uses WordPiece tokenization to handle out-of-vocabulary words.",
        "The [CLS] token is used for classification tasks.",
        "The [SEP] token separates different segments of text.",
        "Position embeddings help the model understand word order.",
        "Self-attention allows tokens to attend to all positions.",
        "Multi-head attention captures different types of relationships.",
        "Layer normalization stabilizes training of deep networks.",
        "Dropout prevents overfitting during training."
    ]
    
    # Duplicate texts to create a larger dataset
    return sample_texts * 50  # 1000 samples
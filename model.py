import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig


class BERTMaskedLM(nn.Module):
    """
    BERT model for Masked Language Modeling (MLM) task.
    This is a wrapper around HuggingFace's BertForMaskedLM with additional functionality.
    """
    
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(model_name)
        self.config = self.bert.config
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tensor of input token IDs
            attention_mask: Tensor indicating which tokens should be attended to
            labels: Tensor of labels for masked tokens (-100 for non-masked tokens)
            
        Returns:
            Dictionary with loss and logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def save_pretrained(self, save_path):
        """Save model and configuration."""
        self.bert.save_pretrained(save_path)
        
    @classmethod
    def from_pretrained(cls, model_path):
        """Load model from saved checkpoint."""
        model = cls.__new__(cls)
        super(BERTMaskedLM, model).__init__()
        model.bert = BertForMaskedLM.from_pretrained(model_path)
        model.config = model.bert.config
        return model
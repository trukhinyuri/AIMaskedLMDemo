import torch
from transformers import BertTokenizer
import argparse
from typing import List, Dict, Tuple
import os

from model import BERTMaskedLM


class BERTInference:
    """Class for performing inference with a trained BERT model."""
    
    def __init__(self, model_path: str, device: str = None, use_default_if_missing: bool = True):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the saved model
            device: Device to run inference on (cuda/cpu/auto)
            use_default_if_missing: Use default BERT model if custom model not found
        """
        if device is None or device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Check if model path exists
        if os.path.exists(os.path.join(model_path, 'config.json')):
            print(f"Loading custom model from {model_path}")
            # Load tokenizer and model from custom path
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BERTMaskedLM.from_pretrained(model_path)
        elif use_default_if_missing:
            print(f"Model not found at {model_path}")
            print("Loading default BERT model (bert-base-uncased)...")
            # Use default BERT model
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BERTMaskedLM('bert-base-uncased')
        else:
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Please train a model first using train.py or use --use_default flag"
            )
        
        self.model.to(self.device)
        self.model.eval()
        
    def predict_masked(self, text: str, top_k: int = 5) -> List[Dict[str, float]]:
        """
        Predict masked tokens in the input text.
        
        Args:
            text: Input text with [MASK] tokens
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions for each [MASK] token
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Find masked positions
        mask_token_id = self.tokenizer.mask_token_id
        masked_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)
        
        if len(masked_positions[0]) == 0:
            return []
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
        
        predictions = []
        
        # Process each masked position
        for batch_idx, token_idx in zip(masked_positions[0], masked_positions[1]):
            # Get top k predictions for this position
            token_logits = logits[batch_idx, token_idx]
            probs = torch.softmax(token_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
            
            # Convert to tokens
            position_predictions = []
            for prob, idx in zip(top_k_probs.cpu(), top_k_indices.cpu()):
                token = self.tokenizer.decode([idx])
                position_predictions.append({
                    'token': token,
                    'probability': float(prob),
                    'token_id': int(idx)
                })
            
            predictions.append(position_predictions)
        
        return predictions
    
    def fill_masks(self, text: str, top_k: int = 1) -> str:
        """
        Fill all [MASK] tokens with the most likely predictions.
        
        Args:
            text: Input text with [MASK] tokens
            top_k: Use the k-th most likely prediction (1 = most likely)
            
        Returns:
            Text with [MASK] tokens replaced
        """
        predictions = self.predict_masked(text, top_k=top_k)
        
        if not predictions:
            return text
        
        # Replace masks with predictions
        result = text
        for pred_list in predictions:
            if pred_list and len(pred_list) >= top_k:
                # Replace first occurrence of [MASK] with prediction
                result = result.replace('[MASK]', pred_list[top_k-1]['token'], 1)
        
        return result
    
    def score_text(self, text: str) -> float:
        """
        Calculate perplexity score for the given text.
        Lower scores indicate better model confidence.
        
        Args:
            text: Input text to score
            
        Returns:
            Perplexity score
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Create labels (same as input_ids, but with padding tokens set to -100)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Calculate loss
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs['loss']
        
        # Convert to perplexity
        perplexity = torch.exp(loss)
        
        return float(perplexity)


def interactive_demo(model_path: str, use_default: bool = True):
    """Run an interactive demo of the model."""
    # Load model
    bert = BERTInference(model_path, use_default_if_missing=use_default)
    
    print("\n" + "="*50)
    print("BERT Masked Language Model - Interactive Demo")
    print("="*50)
    print("\nInstructions:")
    print("- Enter text with [MASK] tokens to predict")
    print("- Type 'quit' to exit")
    print("- Type 'score: <text>' to calculate perplexity")
    print("\nExamples:")
    print("  The [MASK] is shining brightly today.")
    print("  BERT is a [MASK] language [MASK] model.")
    print("\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if text.startswith('score:'):
                # Score mode
                text_to_score = text[6:].strip()
                if text_to_score:
                    score = bert.score_text(text_to_score)
                    print(f"\nPerplexity: {score:.2f}")
                    print("(Lower is better - indicates model confidence)\n")
                continue
            
            if '[MASK]' not in text:
                print("Please include at least one [MASK] token in your text.\n")
                continue
            
            # Get predictions
            predictions = bert.predict_masked(text, top_k=5)
            
            # Display results
            print(f"\nInput: {text}")
            print("\nPredictions:")
            
            for i, pred_list in enumerate(predictions, 1):
                print(f"\n[MASK] #{i}:")
                for j, pred in enumerate(pred_list, 1):
                    print(f"  {j}. {pred['token']:15} ({pred['probability']*100:.1f}%)")
            
            # Show filled text
            filled = bert.fill_masks(text)
            print(f"\nFilled text: {filled}")
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='BERT Masked Language Model Inference')
    parser.add_argument('--model_path', type=str, default='models/best_model',
                        help='Path to the saved model')
    parser.add_argument('--text', type=str, default=None,
                        help='Text with [MASK] tokens (if not provided, runs interactive mode)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help='Device to run inference on')
    parser.add_argument('--use_default', action='store_true',
                        help='Use default BERT model if custom model not found')
    
    args = parser.parse_args()
    
    if args.text:
        # Single prediction mode
        bert = BERTInference(args.model_path, device=args.device, use_default_if_missing=args.use_default)
        
        if '[MASK]' not in args.text:
            print("Error: Text must contain at least one [MASK] token")
            return
        
        predictions = bert.predict_masked(args.text, top_k=args.top_k)
        
        print(f"Input: {args.text}")
        print("\nPredictions:")
        
        for i, pred_list in enumerate(predictions, 1):
            print(f"\n[MASK] #{i}:")
            for j, pred in enumerate(pred_list, 1):
                print(f"  {j}. {pred['token']:15} ({pred['probability']*100:.1f}%)")
        
        filled = bert.fill_masks(args.text)
        print(f"\nFilled text: {filled}")
    else:
        # Interactive mode
        interactive_demo(args.model_path, use_default=args.use_default)


if __name__ == "__main__":
    main()
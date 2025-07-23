import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer
from tqdm import tqdm
import argparse
from datetime import datetime

from model import BERTMaskedLM
from data_loader import create_dataloader, load_sample_data


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            
            # Calculate accuracy on masked tokens
            predictions = torch.argmax(outputs['logits'], dim=-1)
            mask = labels != -100
            
            if mask.sum() > 0:
                correct = (predictions[mask] == labels[mask]).sum().item()
                total_correct += correct
                total_predictions += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train BERT for Masked Language Modeling')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pretrained BERT model name')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--mlm_probability', type=float, default=0.15,
                        help='Probability of masking tokens')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BERTMaskedLM(args.model_name).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load data
    print("\nLoading training data...")
    texts = load_sample_data()
    
    # Split data into train and validation
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_texts,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        shuffle=True
    )
    
    val_dataloader = create_dataloader(
        val_texts,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mlm_probability=args.mlm_probability,
        shuffle=False
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model')
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Best model saved to {save_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}')
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model')
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\nTraining completed!")
    print(f"Final model saved to {final_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
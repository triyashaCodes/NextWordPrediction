import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Import the model and functions from the main script
from pytorch_lstm_attention import (
    LSTMAttentionModel, 
    load_and_preprocess_data,
    create_vocabulary_and_sequences,
    create_data_loaders,
    evaluate_model,
    test_text_generation,
    generate_text
)

def evaluate_model_performance(model_path="sherlock_lstm_attention_pytorch.pth"):
    """Evaluate a saved PyTorch LSTM + Attention model"""
    print("=" * 60)
    print("PYTORCH LSTM + ATTENTION MODEL EVALUATION")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first or provide the correct path.")
        return None
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    text = load_and_preprocess_data()
    
    # Create vocabulary and sequences
    print("\nCreating vocabulary and sequences...")
    vocab, word2idx, idx2word, max_seq_len, start_idx, eos_idx, X_train, X_val, X_test, y_train, y_val, y_test = create_vocabulary_and_sequences(text)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Load saved model
    print(f"\nLoading model from: {model_path}")
    try:
        model = LSTMAttentionModel(len(vocab))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully!")
        print(f"Model parameters: {total_params:,}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    test_acc, test_loss, perplexity = evaluate_model(model, test_loader, device)
    
    # Test text generation
    print("\n" + "=" * 60)
    print("TEXT GENERATION TEST")
    print("=" * 60)
    
    test_prompts = [
        "Holmes is ",
        "Watson was",
        "The detective",
        "I saw",
        "It was a dark",
        "Sherlock Holmes",
        "The case",
        "My dear Watson"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            generated = generate_text(model, prompt, word2idx, idx2word, start_idx, eos_idx, 
                                   max_seq_len, max_length=15, temperature=0.8, device=device)
            print(f"Generated: '{generated}'")
        except Exception as e:
            print(f"Error generating text: {e}")
        print("-" * 40)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Vocabulary Size: {len(vocab):,}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")
    print(f"Model Parameters: {total_params:,}")
    print("=" * 60)
    
    return {
        'model': model,
        'word2idx': word2idx,
        'idx2word': idx2word,
        'start_idx': start_idx,
        'eos_idx': eos_idx,
        'max_seq_len': max_seq_len,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'perplexity': perplexity,
        'vocab_size': len(vocab),
        'total_params': total_params
    }

def interactive_text_generation(model_path="sherlock_lstm_attention_pytorch.pth"):
    """Interactive text generation with the trained model"""
    print("=" * 60)
    print("INTERACTIVE TEXT GENERATION")
    print("=" * 60)
    
    # Load model and data
    result = evaluate_model_performance(model_path)
    if result is None:
        return
    
    model = result['model']
    word2idx = result['word2idx']
    idx2word = result['idx2word']
    start_idx = result['start_idx']
    eos_idx = result['eos_idx']
    max_seq_len = result['max_seq_len']
    device = next(model.parameters()).device
    
    print("\n" + "=" * 60)
    print("INTERACTIVE GENERATION")
    print("=" * 60)
    print("Enter text prompts to generate continuations.")
    print("Type 'quit' to exit.")
    print("-" * 60)
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() == 'quit':
                break
            
            if not prompt:
                print("Please enter a valid prompt.")
                continue
            
            print(f"\nGenerating text for: '{prompt}'")
            generated = generate_text(model, prompt, word2idx, idx2word, start_idx, eos_idx, 
                                   max_seq_len, max_length=20, temperature=0.8, device=device)
            print(f"Generated: '{generated}'")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            # Interactive text generation
            model_path = sys.argv[2] if len(sys.argv) > 2 else "sherlock_lstm_attention_pytorch.pth"
            interactive_text_generation(model_path)
        else:
            # Evaluate specific model
            model_path = sys.argv[1]
            evaluate_model_performance(model_path)
    else:
        # Evaluate default model
        evaluate_model_performance()

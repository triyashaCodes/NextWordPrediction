import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import requests
import os
import re
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# 1. Download & Preprocess Text
# -----------------------------

def download_sherlock_holmes():
    url = "https://www.gutenberg.org/files/1661/1661-0.txt"
    filename = "sherlock_holmes.txt"
    
    if not os.path.exists(filename):
        print("Downloading Sherlock Holmes text...")
        r = requests.get(url, timeout=30)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(r.text)
        print("Downloaded and saved as", filename)
    else:
        print("Using cached file:", filename)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"[''']", "'", text)
    text = re.sub(r'[—–]', '-', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-\(\)]', '', text)
    return text.strip()

def load_and_preprocess_data():
    """Load and preprocess the Sherlock Holmes dataset"""
    print("=" * 60)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 60)
    
    download_sherlock_holmes()

    with open('sherlock_holmes.txt', 'r', encoding='utf-8') as f:
        raw_text = f.read()

    start_idx = raw_text.find("*** START OF")
    end_idx = raw_text.find("*** END OF")
    text = raw_text[start_idx:end_idx] if start_idx != -1 and end_idx != -1 else raw_text
    text = clean_text(text)
    
    print(f"Raw text length: {len(raw_text):,} characters")
    print(f"Cleaned text length: {len(text):,} characters")
    
    return text

# -----------------------------
# 2. Tokenization & Sequence Creation
# -----------------------------

# Add special tokens for sequence control
START_TOKEN = '<START>'
EOS_TOKEN = '<EOS>'

def proper_sentence_split(text):
    """Better sentence splitting that handles punctuation correctly"""
    # Split on sentence-ending punctuation, but keep the punctuation
    sentences = re.split(r'([.!?]+)', text)
    
    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1] if i + 1 < len(sentences) else ''
        if sentence:  # Only add non-empty sentences
            result.append(sentence + punct)
    
    return result

def create_sliding_window_sequences(text, window_size=10):
    """Create overlapping sequences using sliding window approach"""
    tokens = text.split()
    sequences = []
    
    # Create overlapping sequences across the entire text
    for i in range(len(tokens) - window_size + 1):
        sequence = tokens[i:i + window_size]
        sequences.append(sequence)
    
    return sequences

def create_vocabulary_and_sequences(text):
    """Create vocabulary and sequences for training"""
    print("\n" + "=" * 60)
    print("CREATING VOCABULARY AND SEQUENCES")
    print("=" * 60)
    
    print("Creating vocabulary from tokens...")
    tokens = text.split()
    vocab = ['<PAD>', '<UNK>', START_TOKEN, EOS_TOKEN] + sorted(set(tokens))
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}

    # Get indices for special tokens
    start_idx = word2idx[START_TOKEN]
    eos_idx = word2idx[EOS_TOKEN]
    print(f"START token index: {start_idx}, EOS token index: {eos_idx}")
    print(f"Vocabulary size: {len(vocab):,}")

    print("Generating sequences with cross-sentence context...")

    # Method 1: Sliding window across entire text (for cross-sentence learning)
    sliding_sequences = create_sliding_window_sequences(text, window_size=15)

    # Method 2: Proper sentence-based sequences with START/EOS tokens
    sentences = proper_sentence_split(text)
    sentence_sequences = []

    for sentence in sentences:
        sentence_words = sentence.strip().split()
        if len(sentence_words) >= 2:  # Only process sentences with at least 2 words
            # Add START and EOS tokens
            tokenized = [START_TOKEN] + sentence_words + [EOS_TOKEN]
            # Create progressive sequences
            for i in range(2, len(tokenized) + 1):
                sentence_sequences.append(tokenized[:i])

    print(f"Generated {len(sliding_sequences):,} sliding window sequences")
    print(f"Generated {len(sentence_sequences):,} sentence-based sequences")

    # Combine both approaches for richer training data
    all_word_sequences = sliding_sequences + sentence_sequences

    # Convert to indices
    sequences = []
    for seq in all_word_sequences:
        tokenized = [word2idx.get(w, word2idx['<UNK>']) for w in seq]
        sequences.append(tokenized)

    print(f"Total sequences: {len(sequences):,}")
    print("Sample sequences (first 5):")
    for i, seq in enumerate(sequences[:5]):
        words = [idx2word[idx] for idx in seq]
        print(f"  {i+1}: {' '.join(words)}")

    max_seq_len = min(50, max(len(seq) for seq in sequences))
    sequences = [([0] * (max_seq_len - len(seq)) + seq)[-max_seq_len:] for seq in sequences]

    sequences = np.array(sequences)
    X, y = sequences[:, :-1], sequences[:, -1]
    # Keep y as label encoding (integer labels) instead of one-hot encoding
    y = torch.tensor(y).long()

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)

    print(f"\nDATA SHAPES:")
    print(f"   • Vocabulary size: {len(vocab):,}")
    print(f"   • Max sequence length: {max_seq_len}")
    print(f"   • Training samples: {len(X_train):,}")
    print(f"   • Validation samples: {len(X_val):,}")
    print(f"   • Test samples: {len(X_test):,}")

    return vocab, word2idx, idx2word, max_seq_len, start_idx, eos_idx, X_train, X_val, X_test, y_train, y_val, y_test

# -----------------------------
# 3. Dataset & DataLoader
# -----------------------------

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).long()
        self.y = torch.tensor(y).long() if not isinstance(y, torch.Tensor) else y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
    """Create data loaders for training, validation, and test sets"""
    print("\n" + "=" * 60)
    print("CREATING DATA LOADERS")
    print("=" * 60)
    
    train_loader = DataLoader(TextDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TextDataset(X_test, y_test), batch_size=batch_size)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

# -----------------------------
# 4. Model Definition
# -----------------------------

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attention_units):
        super(AttentionLayer, self).__init__()
        self.attention_dense = nn.Linear(hidden_dim, attention_units)
        self.context_vector = nn.Linear(attention_units, 1, bias=False)

    def forward(self, lstm_out):
        score = torch.tanh(self.attention_dense(lstm_out))
        attention_weights = torch.softmax(self.context_vector(score), dim=1)
        context_vector = attention_weights * lstm_out
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights.squeeze(-1)

class LSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, lstm_units=100, attention_units=64):
        super(LSTMAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.attention = AttentionLayer(lstm_units, attention_units)
        self.layer_norm = nn.LayerNorm(lstm_units)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_units, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.layer_norm(x)
        context, attention_weights = self.attention(x)
        out = self.dropout(context)
        return self.fc(out), attention_weights

def build_model(vocab_size, embedding_dim=100, lstm_units=100, attention_units=64):
    """Build the LSTM Attention model"""
    print("\n" + "=" * 60)
    print("BUILDING LSTM ATTENTION MODEL")
    print("=" * 60)
    
    model = LSTMAttentionModel(vocab_size, embedding_dim, lstm_units, attention_units)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture:")
    print(f"   • Vocabulary size: {vocab_size:,}")
    print(f"   • Embedding dimension: {embedding_dim}")
    print(f"   • LSTM units: {lstm_units}")
    print(f"   • Attention units: {attention_units}")
    print(f"   • Total parameters: {total_params:,}")
    print(f"   • Trainable parameters: {trainable_params:,}")
    
    return model

# -----------------------------
# 5. Training Loop
# -----------------------------

import torch.nn.utils as nn_utils

def train_model(model, train_loader, val_loader, device, epochs=100, learning_rate=0.005):
    """Train the LSTM Attention model"""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping: clip gradients to max norm 1.0
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += inputs.size(0)
        
        train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                val_total += inputs.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_lstm_attention_model.pth")
            print(f"New best model saved! (Val Acc: {val_acc:.4f})")

        # Step scheduler with validation loss
        scheduler.step(val_loss)
        
        # Early stopping (optional)
        if epoch > 20 and val_loss > min(val_losses[-5:]):
            print("Early stopping triggered")
            break

    return train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('lstm_attention_training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved: lstm_attention_training_history.png")

# -----------------------------
# 6. Text Generation
# -----------------------------

def generate_text(model, seed_text, word2idx, idx2word, start_idx, eos_idx, max_seq_len, 
                 max_length=50, temperature=1.0, device='cpu'):
    """
    Generate text from a seed text by iteratively predicting next tokens.
    For each predicted token, print the top 5 most likely next tokens with probabilities.

    Args:
        model: Trained LSTM attention model
        seed_text (str): Initial text prompt to start generation
        max_length (int): Maximum length of generated tokens (including seed)
        temperature (float): Sampling temperature for controlling randomness

    Returns:
        str: Generated text sequence including the seed_text
    """
    model.eval()
    
    # Tokenize seed text
    tokens = seed_text.lower().split()
    sequence = [start_idx]  # Start with <START> token
    
    # Map seed tokens to indices (using <UNK> if not found)
    for token in tokens:
        sequence.append(word2idx.get(token, word2idx['<UNK>']))
    
    generated_tokens = tokens.copy()

    with torch.no_grad():
        for _ in range(max_length - len(sequence) + 1):  # Adjust for seed length
            # Prepare input: last max_seq_len tokens, padded if needed
            if len(sequence) < max_seq_len:
                padding = [0] * (max_seq_len - len(sequence))
                input_seq = torch.tensor([padding + sequence]).to(device)
            else:
                input_seq = torch.tensor([sequence[-max_seq_len:]]).to(device)

            # Get model output logits
            output, _ = model(input_seq)
            logits = output[0] / temperature
            probs = torch.softmax(logits, dim=0)

            # Get top 5 predicted tokens and probabilities
            top5_probs, top5_idx = torch.topk(probs, 5)
            top5_words = [idx2word[idx.item()] for idx in top5_idx]

            # Print top 5 predictions with probabilities
            print(f"Top 5 next words: {[(w, float(p)) for w, p in zip(top5_words, top5_probs)]}")

            # Sample next token from probability distribution
            next_token_idx = torch.multinomial(probs, 1).item()

            # Stop if EOS token generated
            if next_token_idx == eos_idx:
                break

            # Append predicted token to sequence and generated output tokens
            sequence.append(next_token_idx)
            next_word = idx2word.get(next_token_idx, '<UNK>')

            # Avoid adding special tokens to generated text output
            if next_word not in ['<PAD>', '<UNK>', START_TOKEN]:
                generated_tokens.append(next_word)

    return ' '.join(generated_tokens)

def test_text_generation(model, word2idx, idx2word, start_idx, eos_idx, max_seq_len, device):
    """Test text generation with various prompts"""
    print("\n" + "=" * 60)
    print("TESTING TEXT GENERATION")
    print("=" * 60)
    
    test_prompts = [
        "Holmes is ",
        "Watson was",
        "The detective",
        "I saw",
        "It was a dark"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = generate_text(model, prompt, word2idx, idx2word, start_idx, eos_idx, 
                                max_seq_len, max_length=20, temperature=0.8, device=device)
        print(f"Generated: '{generated}'")
        print("-" * 40)

# -----------------------------
# 7. Model Evaluation
# -----------------------------

def calculate_perplexity(model, test_loader, device):
    """Calculate perplexity on test data"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            
            # Calculate cross-entropy loss
            loss = nn.CrossEntropyLoss(reduction='sum')(outputs, labels)
            total_loss += loss.item()
            total_samples += inputs.size(0)
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return float(perplexity), avg_loss

def evaluate_model(model, test_loader, device):
    """Evaluate the model on test data"""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    model.eval()
    test_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += inputs.size(0)
    
    test_loss /= total
    test_acc = correct / total
    
    # Calculate perplexity
    perplexity, avg_loss = calculate_perplexity(model, test_loader, device)
    
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")
    print(f"Average Cross-Entropy Loss: {avg_loss:.4f}")
    
    return test_acc, test_loss, perplexity

def load_saved_model(model_path, vocab_size, device):
    """Load a saved model from .pth file"""
    print("\n" + "=" * 60)
    print("LOADING SAVED MODEL")
    print("=" * 60)
    
    try:
        # Create model with same architecture
        model = LSTMAttentionModel(vocab_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully from: {model_path}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# -----------------------------
# 8. Main Pipeline
# -----------------------------

def main():
    """Main training pipeline"""    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    text = load_and_preprocess_data()
    
    # Create vocabulary and sequences
    vocab, word2idx, idx2word, max_seq_len, start_idx, eos_idx, X_train, X_val, X_test, y_train, y_val, y_test = create_vocabulary_and_sequences(text)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Build model
    model = build_model(len(vocab))
    model = model.to(device)
    
    # Train model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, val_loader, device)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # Evaluate model
    test_acc, test_loss, perplexity = evaluate_model(model, test_loader, device)
    
    # Test text generation
    test_text_generation(model, word2idx, idx2word, start_idx, eos_idx, max_seq_len, device)
    
    # Save final model
    torch.save(model.state_dict(), "sherlock_lstm_attention_pytorch.pth")
    print("\nModel saved as 'sherlock_lstm_attention_pytorch.pth'")
    
    return model, word2idx, idx2word, start_idx, eos_idx, max_seq_len

def evaluate_saved_model(model_path, data_path="sherlock_holmes.txt"):
    """Evaluate a saved model without training"""
    print("=" * 60)
    print("EVALUATING SAVED MODEL")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    text = load_and_preprocess_data()
    
    # Create vocabulary and sequences
    vocab, word2idx, idx2word, max_seq_len, start_idx, eos_idx, X_train, X_val, X_test, y_train, y_val, y_test = create_vocabulary_and_sequences(text)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Load saved model
    model = load_saved_model(model_path, len(vocab), device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Evaluate model
    test_acc, test_loss, perplexity = evaluate_model(model, test_loader, device)
    
    # Test text generation
    test_text_generation(model, word2idx, idx2word, start_idx, eos_idx, max_seq_len, device)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")
    print("=" * 60)
    
    return model, word2idx, idx2word, start_idx, eos_idx, max_seq_len

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        # Evaluate a saved model
        model_path = sys.argv[2] if len(sys.argv) > 2 else "sherlock_lstm_attention_pytorch.pth"
        print(f"Evaluating saved model: {model_path}")
        model, word2idx, idx2word, start_idx, eos_idx, max_seq_len = evaluate_saved_model(model_path)
    else:
        # Train a new model
        print("Training new model...")
        model, word2idx, idx2word, start_idx, eos_idx, max_seq_len = main()
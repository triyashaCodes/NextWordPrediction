import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import os

class BasicLSTMModel:
    def __init__(self, embedding_dim=100, lstm_units=100):
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.tokenizer = None
        self.model = None
        self.history = None
        
    def load_data(self, file_path='sherlock_holmes_cleaned.txt'):
        """Load and prepare the cleaned Sherlock Holmes dataset"""
        print("=" * 60)
        print("LOADING AND PREPARING DATASET")
        print("=" * 60)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into lines and filter out empty lines
            corpus = [line.strip() for line in content.split('\n') if len(line.strip()) > 10]
            
            print(f"✓ Dataset loaded: {len(corpus)} lines")
            print(f"✓ Total characters: {len(content):,}")
            
            return corpus
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, corpus):
        """Preprocess the corpus for next word prediction"""
        print("\n" + "=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        
        # 1. Initialize tokenizer and fit on texts
        print("1. Initializing tokenizer...")
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(corpus)
        
        # Vocabulary size (+1 for padding token)
        total_words = len(self.tokenizer.word_index) + 1
        print(f"✓ Vocabulary size: {total_words:,}")
        
        # 2. Create input sequences for next word prediction
        print("2. Creating input sequences...")
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_seq = token_list[:i+1]
                input_sequences.append(n_gram_seq)
        
        print(f"✓ Created {len(input_sequences):,} sequences")
        
        # 3. Pad sequences to have uniform length
        print("3. Padding sequences...")
        max_seq_len = max(len(seq) for seq in input_sequences)
        input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
        
        print(f"✓ Max sequence length: {max_seq_len}")
        
        # 4. Split predictors and labels
        print("4. Splitting predictors and labels...")
        predictors = input_sequences[:, :-1]
        labels = input_sequences[:, -1]
        
        # 5. One-hot encode labels
        print("5. One-hot encoding labels...")
        labels = to_categorical(labels, num_classes=total_words)
        
        print("\n📊 DATA SHAPES:")
        print(f"   • Total words (vocab size): {total_words:,}")
        print(f"   • Max sequence length: {max_seq_len}")
        print(f"   • Predictors shape: {predictors.shape}")
        print(f"   • Labels shape: {labels.shape}")
        
        return predictors, labels, total_words, max_seq_len
    
    def build_model(self, total_words, max_seq_len):
        """Build the basic LSTM model with reduced complexity and increased regularization"""
        print("\n" + "=" * 60)
        print("BUILDING TWEAKED BASIC LSTM MODEL")
        print("=" * 60)
        
        self.model = Sequential([
            Embedding(input_dim=total_words, 
                    output_dim=self.embedding_dim, 
                    input_length=max_seq_len - 1),
            LSTM(64,  # Reduced units
                return_sequences=False,
                dropout=0.3,           # Increased dropout inside LSTM
                recurrent_dropout=0.2),# Recurrent dropout added
            Dropout(0.3),               # Increased dropout after LSTM
            Dense(total_words, activation='softmax')
        ])
        
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print("Model Summary:")
        self.model.summary()
        
        return self.model

    def train_model(self, predictors, labels, epochs=30, batch_size=64, validation_split=0.1):
        """Train the model"""
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ModelCheckpoint('basic_lstm_best.keras', 
                        monitor='val_accuracy', 
                        save_best_only=True, 
                        verbose=1)
        ]
        
        print(f"Training parameters:")
        print(f"   • Epochs: {epochs}")
        print(f"   • Batch size: {batch_size}")
        print(f"   • Validation split: {validation_split}")
        print(f"   • Total samples: {len(predictors):,}")
        
        self.history = self.model.fit(
            predictors, 
            labels, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history

    def predict_next_word(self, input_text, top_k=5):
        """Predict the next word given an input sequence"""
        if self.model is None or self.tokenizer is None:
            print("Model or tokenizer not loaded")
            return None
        
        try:
            token_list = self.tokenizer.texts_to_sequences([input_text])[0]
            
            if len(token_list) == 0:
                print(f"Input text '{input_text}' contains no known tokens.")
                return None, None
            
            max_seq_len = self.model.input_shape[1] + 1
            padded_sequence = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
            input_sequence = padded_sequence[:, :-1]
            
            predictions = self.model.predict(input_sequence, verbose=0)[0]
            
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            index_to_word = self.tokenizer.index_word  # use tokenizer's index_word dict
            
            print(f"\nPredictions for: '{input_text}'")
            print("-" * 40)
            for i, idx in enumerate(top_indices):
                word = index_to_word.get(idx, f"<UNK_{idx}>")
                probability = predictions[idx]
                print(f"{i+1}. '{word}' - Probability: {probability:.4f} ({probability*100:.2f}%)")
            
            return top_indices, predictions[top_indices]
            
        except Exception as e:
            print(f"Error predicting next word: {e}")
            return None, None

    
    def save_model_and_tokenizer(self):
        """Save the trained model and tokenizer"""
        print("\n" + "=" * 60)
        print("SAVING MODEL AND TOKENIZER")
        print("=" * 60)
        
        # Save model
        self.model.save('basic_lstm_model.keras')
        print("✓ Model saved: basic_lstm_model.keras")
        
        # Save tokenizer
        with open('basic_lstm_tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("✓ Tokenizer saved: basic_lstm_tokenizer.pickle")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('basic_lstm_training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history plot saved: basic_lstm_training_history.png")
    
    
    def evaluate_model(self, predictors, labels):
        """Evaluate the model on test data"""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Split data for evaluation (use last 20% as test set)
        split_idx = int(len(predictors) * 0.8)
        test_predictors = predictors[split_idx:]
        test_labels = labels[split_idx:]
        
        print(f"Test set size: {len(test_predictors):,} samples")
        
        # Evaluate
        test_loss, test_accuracy = self.model.evaluate(test_predictors, test_labels, verbose=0)
        
        print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"✓ Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 STARTING BASIC LSTM TRAINING PIPELINE")
        
        # Load data
        corpus = self.load_data()
        if corpus is None:
            return False
        
        # Preprocess data
        predictors, labels, total_words, max_seq_len = self.preprocess_data(corpus)
        
        # Build model
        self.build_model(total_words, max_seq_len)
        
        # Train model
        self.train_model(predictors, labels)
        
        # Save model and tokenizer
        self.save_model_and_tokenizer()
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate model
        self.evaluate_model(predictors, labels)
        
        # Test predictions
        print("\n" + "=" * 60)
        print("TESTING PREDICTIONS")
        print("=" * 60)
        
        test_phrases = [
            "Sherlock Holmes",
            "Watson said",
            "The case",
            "I am",
            "Holmes is"
        ]
        
        for phrase in test_phrases:
            self.predict_next_word(phrase, top_k=3)
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print("=" * 60)
        print("Files created:")
        print("   • basic_lstm_model.keras - Trained model")
        print("   • basic_lstm_tokenizer.pickle - Tokenizer")
        print("   • basic_lstm_best.keras - Best model checkpoint")
        print("   • basic_lstm_training_history.png - Training plots")
        
        return True

if __name__ == "__main__":
    # Create and run the basic LSTM model
    lstm_model = BasicLSTMModel(embedding_dim=100, lstm_units=100)
    lstm_model.run_complete_pipeline()

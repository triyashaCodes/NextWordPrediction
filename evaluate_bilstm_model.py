import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle
import math

def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model('model_ABiLSTM.keras')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return None, None

def load_test_data(file_path='sherlock_holmes.txt'):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus = f.read().split('\n')
        corpus = [line.strip() for line in corpus if line.strip()]
        test_size = int(len(corpus) * 0.2)
        return corpus[-test_size:]
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def prepare_test_sequences(test_corpus, tokenizer, max_seq_len):
    input_sequences = []
    actual_words = []
    for line in test_corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)
            actual_words.append(token_list[i])
    padded_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
    predictors = padded_sequences[:, :-1]
    labels = to_categorical(padded_sequences[:, -1], num_classes=len(tokenizer.word_index) + 1)
    return predictors, labels, actual_words

def calculate_accuracy(model, predictors, labels):
    try:
        predictions = model.predict(predictors, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(labels, axis=1)
        correct = np.sum(predicted_classes == actual_classes)
        total = len(actual_classes)
        return correct / total, correct, total
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        return None, None, None

def calculate_perplexity(model, predictors, labels):
    try:
        predictions = model.predict(predictors, verbose=0)
        epsilon = 1e-10
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        log_probs = np.log(predictions)
        actual_probs = np.sum(log_probs * labels, axis=1)
        avg_neg_log_likelihood = -np.mean(actual_probs)
        perplexity = math.exp(avg_neg_log_likelihood)
        return perplexity, avg_neg_log_likelihood
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None, None

def predict_next_word(model, tokenizer, input_text, top_k=5):
    try:
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        max_seq_len = model.input_shape[1] + 1
        padded_sequence = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')
        input_sequence = padded_sequence[:, :-1]
        predictions = model.predict(input_sequence, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        index_to_word = {v: k for k, v in tokenizer.word_index.items()}
        print(f"\nPredictions for: '{input_text}'")
        for i, idx in enumerate(top_indices):
            word = index_to_word.get(idx, f"<UNK_{idx}>")
            probability = predictions[idx]
            print(f"{i+1}. '{word}' - Probability: {probability:.4f}")
        return top_indices, predictions[top_indices]
    except Exception as e:
        print(f"Error predicting next word: {e}")
        return None, None

def evaluate_model():
    print("=" * 60)
    print("BiLSTM Model Evaluation")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        return
    test_corpus = load_test_data()
    if test_corpus is None:
        return
    max_seq_len = model.input_shape[1] + 1
    total_words = len(tokenizer.word_index) + 1
    print(f"\nModel Parameters:")
    print(f"- Max sequence length: {max_seq_len}")
    print(f"- Vocabulary size: {total_words}")
    print(f"- Model input shape: {model.input_shape}")
    print(f"- Model output shape: {model.output_shape}")
    predictors, labels, actual_words = prepare_test_sequences(test_corpus, tokenizer, max_seq_len)
    print(f"Test sequences: {len(predictors)}")
    accuracy, correct, total = calculate_accuracy(model, predictors, labels)
    if accuracy is not None:
        print(f"Test Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    perplexity, avg_neg_log_likelihood = calculate_perplexity(model, predictors, labels)
    if perplexity is not None:
        print(f"Test Perplexity: {perplexity:.4f}")
    print("\n" + "=" * 60)
    print("SPECIFIC PREDICTIONS")
    print("=" * 60)
    predict_next_word(model, tokenizer, "Holmes is a", top_k=5)
    test_phrases = [
        "Sherlock Holmes",
        "Watson said",
        "The case",
        "I am"
    ]
    for phrase in test_phrases:
        predict_next_word(model, tokenizer, phrase, top_k=3)
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if accuracy is not None:
        print(f"Test Accuracy: {accuracy*100:.2f}%")
    if perplexity is not None:
        print(f"Test Perplexity: {perplexity:.4f}")
    print(f"Test Sequences: {len(predictors)}")
    print(f"Vocabulary Size: {total_words}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_model()

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Bidirectional, Dense, Attention, Concatenate, Dropout
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam




# --------- Step 1: Load Dataset ---------
def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        corpus = f.read().split('\n')
    corpus = [line.strip() for line in corpus if len(line.strip()) > 0]
    return corpus

# Replace 'afaan_oromo_corpus.txt' with your dataset file path
corpus = load_corpus('sherlock_holmes.txt')

# --------- Step 2: Preprocess Data ---------
def preprocess_data(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    max_seq_len = max(len(seq) for seq in input_sequences)
    input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

    predictors = input_sequences[:, :-1]
    labels = to_categorical(input_sequences[:, -1], num_classes=total_words)

    return predictors, labels, max_seq_len, total_words, tokenizer

predictors, labels, max_seq_len, total_words, tokenizer = preprocess_data(corpus)
print("data preprocessed")
X_train, X_test, y_train, y_test = train_test_split(
    predictors, labels, test_size=0.1, random_state=42
)

# --------- Step 3: Build Model ---------
def build_attention_bilstm_model(total_words, max_seq_len, embedding_dim=100, lstm_units=100):
    #This is implemented exactly as in the paper: https://bctjournal.com/article_405_d2981e8edce988498ceadcf028429501.pdf
    input_layer = Input(shape=(max_seq_len - 1,), name="Input_Sequence")
    embedding = Embedding(input_dim=total_words, output_dim=embedding_dim, name="Embedding_Layer")(input_layer)
    bi_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True), name="Bidirectional_LSTM")(embedding)

    query = LSTM(lstm_units, return_sequences=True, name="Query_LSTM")(bi_lstm)
    key = LSTM(lstm_units, return_sequences=True, name="Key_LSTM")(bi_lstm)
    value = LSTM(lstm_units, return_sequences=True, name="Value_LSTM")(bi_lstm)

    attention_output = Attention(name="Attention_Layer")([query, value])
    dropout = Dropout(0.3)(attention_output)

    concat = Concatenate(name="Concat_Attn_LSTM")([dropout, bi_lstm])

    final_lstm = LSTM(lstm_units, name="Final_LSTM")(concat)
    output_layer = Dense(total_words, activation='softmax', name="Softmax_Output")(final_lstm)

    model = Model(inputs=input_layer, outputs=output_layer, name="ABiLSTM_NextWordPredictor")
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), metrics=['accuracy'])
    return model

model = build_attention_bilstm_model(total_words, max_seq_len)

# --------- Step 4: Train Model ---------
model.fit(X_train, y_train, epochs=30, verbose=1, batch_size=64,)

# --------- Step 5: Save Model ---------
model.save('model_ABiLSTM.keras')

# --------- Optional: Save tokenizer for later use ---------
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Training complete and model saved!")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Sherlock Holmes Next Word Prediction with LSTM + Attention

## Overview
This project implements a next-word prediction model using LSTM with attention mechanism on the Sherlock Holmes corpus. The model predicts the next word given a sequence of previous words, demonstrating language modeling capabilities.

## Project Structure
```
next_word/
├── pytorch_lstm_attention.py          # Main PyTorch LSTM + Attention model
├── basic_lstm_model.py               # Basic LSTM model (TensorFlow)
├── bilstm_paper.py                   # BiLSTM + Attention model (TensorFlow)
├── data_download_and_analysis.py     # Data preprocessing and analysis
├── sherlock_holmes_cleaned.txt       # Cleaned dataset
├── sherlock_holmes_gutenberg.txt     # Raw dataset
├── pyproject.toml                    # Poetry dependencies
└── README.md                         # This file
```

## Dependencies

### Poetry Configuration
```toml
[tool.poetry]
name = "sherlock-holmes-lstm"
version = "0.1.0"
description = "LSTM + Attention model for next word prediction on Sherlock Holmes corpus"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.8"
torch = "^2.0.0"
tensorflow = "^2.10.0"
numpy = "^1.21.0"
matplotlib = "^3.5.0"
seaborn = "^0.11.0"
pandas = "^1.3.0"
scikit-learn = "^1.0.0"
requests = "^2.25.0"
nltk = "^3.6.0"
wordcloud = "^1.8.0"
tqdm = "^4.62.0"

[tool.poetry.dev-dependencies]
jupyter = "^1.0.0"
ipykernel = "^6.0.0"
```

### Installation
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Usage

### Training Models
```bash
# Train PyTorch LSTM + Attention model
python pytorch_lstm_attention.py

# Train Basic LSTM model (TensorFlow)
python basic_lstm_model.py

# Train BiLSTM + Attention model (TensorFlow)
python bilstm_paper.py
```

### Data Analysis
```bash
# Download and analyze dataset
python data_download_and_analysis.py
```

### Model Training and Evaluation
For detailed information on how the models were trained and how to load and evaluate them, see the **Experimentations** section below. The training scripts include built-in evaluation capabilities and model loading functions.

**Quick Training Commands:**
```bash
# Train PyTorch LSTM + Attention model (includes evaluation)
python pytorch_lstm_attention.py

# Train Basic LSTM model (TensorFlow)
python basic_lstm_model.py

# Train BiLSTM + Attention model (TensorFlow)
python bilstm_paper.py
```

## Approach and Architecture

### Initial Approach: Simple LSTM + Attention
I tried out various approches and began with a basic LSTM + attention architecture:
- Single LSTM layer with attention mechanism
- Vocabulary size: ~8,000 words
- Limited regularization

**Issues Encountered:**
- Small corpus size (102,069 words) led to severe overfitting
- Model memorized training data instead of learning patterns
- Poor generalization on test set

### Final Architecture: Stacked LSTM + Attention with Regularization

#### PyTorch Model Architecture
```python
class LSTMAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, lstm_units=100, attention_units=64):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, lstm_units, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units, batch_first=True)
        self.attention = AttentionLayer(lstm_units, attention_units)
        self.layer_norm = nn.LayerNorm(lstm_units)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_units, vocab_size)
```

#### TensorFlow BiLSTM + Attention Architecture
```python
def build_attention_bilstm_model(total_words, max_seq_len, embedding_dim=100, lstm_units=100):
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
```

#### Attention Mechanism
**Type:** Additive Attention (Bahdanau-style)
```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, attention_units):
        self.attention_dense = nn.Linear(hidden_dim, attention_units)
        self.context_vector = nn.Linear(attention_units, 1, bias=False)

    def forward(self, lstm_out):
        score = torch.tanh(self.attention_dense(lstm_out))
        attention_weights = torch.softmax(self.context_vector(score), dim=1)
        context_vector = attention_weights * lstm_out
        return context_vector, attention_weights
```

**Justification for Additive Attention:**
- Better for variable-length sequences
- More stable training compared to multiplicative attention
- Allows model to focus on relevant parts of input sequence
- Effective for next-word prediction where context matters

**BiLSTM Advantages:**
- Captures bidirectional context (past and future information)
- Better understanding of word dependencies in both directions
- Improved performance on language modeling tasks
- Combines forward and backward LSTM outputs for richer representations
- Instead of considering only the final hidden layer output, we consider the LSTM output at each timestep into a space where the model can learn which parts (timesteps) are important by scoring them differently. This is the Bahdanau attention on LSTM outputs.

## Experimentations

### Training Process and Model Loading

The training scripts include comprehensive evaluation capabilities and model loading functions. Here's how to work with the trained models:

#### PyTorch LSTM + Attention Model

**Training with Evaluation:**
```bash
python pytorch_lstm_attention.py
```

**Key Features:**
- **Built-in Evaluation**: The script automatically evaluates the model on test data
- **Model Saving**: Trained model is saved as `sherlock_lstm_attention_pytorch.pth`
- **Text Generation**: Includes interactive text generation capabilities
- **Vocabulary Handling**: Automatically handles vocabulary size mismatches

**Model Loading and Evaluation:**
The training script includes functions to load and evaluate saved models:
- `load_saved_model()`: Loads trained model with vocabulary size handling
- `evaluate_saved_model()`: Comprehensive evaluation on test data
- `generate_text()`: Interactive text generation

**Example Usage:**
```python
# Load and evaluate model
from pytorch_lstm_attention import load_saved_model, evaluate_saved_model

# Load model (handles vocabulary size mismatches automatically)
model, vocab, word2idx, idx2word = load_saved_model("sherlock_lstm_attention_pytorch.pth")

# Evaluate model
test_acc, test_loss, perplexity = evaluate_saved_model("sherlock_lstm_attention_pytorch.pth")
```

#### TensorFlow BiLSTM + Attention Model

**Training:**
```bash
python bilstm_paper.py
```

**Key Features:**
- **Automatic Model Saving**: Saves model as `model_ABiLSTM.keras`
- **Tokenizer Persistence**: Saves tokenizer as `tokenizer.pickle`
- **Built-in Evaluation**: Includes evaluation metrics during training
- **Text Generation**: Provides next-word prediction capabilities

**Model Loading:**
```python
import tensorflow as tf
import pickle

# Load model
model = tf.keras.models.load_model('model_ABiLSTM.keras')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```

#### Basic LSTM Model (TensorFlow)

**Training:**
```bash
python basic_lstm_model.py
```

**Features:**
- Simple LSTM architecture for baseline comparison
- Automatic model saving and loading
- Built-in evaluation metrics

### Evaluation Metrics

All models provide comprehensive evaluation including:
- **Accuracy**: Top-1 prediction accuracy
- **Perplexity**: Measure of model uncertainty
- **Loss**: Cross-entropy loss on test data
- **Text Generation**: Sample predictions and generated text

### Model Comparison

| Model | Framework | Accuracy | Perplexity | Parameters |
|-------|-----------|----------|------------|------------|
| PyTorch LSTM + Attention | PyTorch | 14.29% | 13.66 | 2.7M |
| BiLSTM + Attention | TensorFlow | 42.52% | 13.66 | ~2M |
| Basic LSTM | TensorFlow | Baseline | Baseline | ~1M |

### Key Findings from Experimentations

1. **Vocabulary Size Impact**: Models trained with different vocabulary sizes show performance variations
2. **Framework Differences**: PyTorch offers better control, TensorFlow provides better performance for this specific task
3. **Attention Effectiveness**: Attention mechanisms significantly improve context understanding
4. **Regularization Importance**: Dropout and normalization crucial for small datasets
5. **Model Loading Robustness**: Automatic vocabulary size handling prevents loading errors

#### Regularization Techniques
1. **Dropout (0.3)**: Prevents overfitting by randomly zeroing activations
2. **Layer Normalization**: Stabilizes training and improves convergence
3. **Gradient Clipping**: Prevents exploding gradients
4. **Label Smoothing (0.1)**: Improves generalization
5. **Early Stopping**: Prevents overfitting by monitoring validation loss

### Corpus Size Justification
The Sherlock Holmes corpus contains only 102,069 words, which is extremely small for deep learning language models. All the research showing 80% accuracy has been on data which were on far bigger files.

**Solutions Implemented to solve for next word prediction:**
- Increased regularization to combat overfitting
- Stacked LSTM layers for better feature extraction
- Attention mechanism to focus on relevant context
- Sliding window + sentence-based sequence creation for more training data

### BiLSTM Implementation Details

The BiLSTM + Attention model uses a sophisticated architecture:

1. **Bidirectional LSTM**: Processes sequences in both forward and backward directions
2. **Query-Key-Value Attention**: Three separate LSTM layers create query, key, and value representations
3. **Attention Mechanism**: Computes attention weights over the sequence
4. **Concatenation**: Combines attention output with original BiLSTM output
5. **Final LSTM**: Processes the concatenated representation for final prediction

This architecture allows the model to:
- Capture dependencies in both directions (past and future context)
- Focus on relevant parts of the input sequence through attention
- Combine global (attention) and local (BiLSTM) information
- Achieve better performance than unidirectional models

## Final Metrics

### PyTorch LSTM + Attention Model
- **Test Accuracy**: 14.29%
- **Test Perplexity**: 13.66
- **Test Loss**: 5.37
- **Vocabulary Size**: 12,715 words
- **Model Parameters**: 2,724,043

### TensorFlow BiLSTM + Attention Model
- **Test Accuracy**: 42.52%
- **Test Perplexity**: 13.66
- **Vocabulary Size**: 8,922 words
- **Model Parameters**: ~2M
- **Architecture**: Bidirectional LSTM with Query-Key-Value attention
- **Key Features**: 
  - Bidirectional context processing
  - Separate Query, Key, Value LSTM layers
  - Attention mechanism on LSTM outputs
  - Concatenation of attention and BiLSTM outputs


## Generated Examples

### PyTorch LSTM + Attention Model Predictions
```
Top 5 next words: [('a', 0.5823346972465515), ('the', 0.066647469997406), ('that', 0.045102473348379135), ('an', 0.03783230856060982), ('very', 0.031026897951960564)]
Top 5 next words: [('very', 0.26571258902549744), ('little', 0.19101107120513916), ('small', 0.12731096148490906), ('man', 0.08128730207681656), ('fierce', 0.0749133825302124)]
Top 5 next words: [('heavy', 0.22029222548007965), ('pretty', 0.19918301701545715), ('little', 0.19044387340545654), ('large', 0.06172473356127739), ('very', 0.037800583988428116)]
Top 5 next words: [('and', 0.8441948294639587), ('between', 0.062399979680776596), ('sleeper,', 0.04337615519762039), ('which', 0.01718759350478649), ('with', 0.007617570459842682)]
Top 5 next words: [('darkness', 0.13406139612197876), ('walked', 0.10114217549562454), ('heavy', 0.06598377972841263), ('held', 0.018099963665008545), ('iron', 0.017817718908190727)]
Top 5 next words: [('the', 0.4372996985912323), ('a', 0.1787565052509308), ('his', 0.04859798401594162), ('him', 0.04081470146775246), ('it', 0.03885069862008095)]
Top 5 next words: [('little', 0.38716986775398254), ('of', 0.12744879722595215), ('small', 0.04793544113636017), ('link', 0.027646662667393684), ('man', 0.02360580489039421)]
Top 5 next words: [('as', 0.14678190648555756), ('there', 0.050766926258802414), ('through', 0.045603539794683456), ('a', 0.039651889353990555), ('so', 0.038531530648469925)]
Top 5 next words: [('a', 0.4313310384750366), ('one', 0.13898111879825592), ('the', 0.060325492173433304), ('to', 0.05426760017871857), ('his', 0.053641095757484436)]
Top 5 next words: [('goose', 0.19888675212860107), ('little', 0.059813980013132095), ('white', 0.057951152324676514), ('link', 0.049473222345113754), ('man', 0.049387238919734955)]
Top 5 next words: [('goose', 0.5769937038421631), ('bent', 0.1453474909067154), ('brown', 0.06134522706270218), ('cheeks', 0.01407970953732729), ('link', 0.01229474414139986)]
Top 5 next words: [('slung', 0.5461320281028748), ('in', 0.17337967455387115), ('upon', 0.14123116433620453), ('of', 0.05041402950882912), ('to', 0.014478065073490143)]
Top 5 next words: [('his', 0.49084237217903137), ('the', 0.3375875651836395), ('slung', 0.021232405677437782), ('all', 0.012349463999271393), ('a', 0.011719029396772385)]
Top 5 next words: [('face', 0.16526374220848083), ('cheeks', 0.0711536556482315), ('voice', 0.046916086226701736), ('shoulder.', 0.040754564106464386), ('head', 0.036930229514837265)]
Top 5 next words: [('head.', 0.9959756731987), ('arm', 0.0016433369601145387), ('impression', 0.00033926410833373666), ('table,', 0.0002466535079292953), ('and', 0.00019402198086027056)]
Top 5 next words: [('<EOS>', 0.9986769556999207), ('of', 7.104963879100978e-05), ('head.', 1.3238919564173557e-05), ('little', 1.3058602235105354e-05), ('chink', 9.392278116138186e-06)]
Prompt: 'Holmes is '
Generated: 'holmes is a very heavy and opened some close-fitting as a white goose upon his flaming head.'
----------------------------------------
Top 5 next words: [('so', 0.10546635091304779), ('but', 0.08663862943649292), ('it', 0.08341487497091293), ('in', 0.06048685684800148), ('not', 0.05543135106563568)]
Top 5 next words: [('the', 0.4040200710296631), ('that', 0.11268283426761627), ('one', 0.07684382051229477), ('it', 0.039939556270837784), ('he', 0.02685757912695408)]
Top 5 next words: [('moment', 0.600499153137207), ('<UNK>', 0.04777751490473747), ('first', 0.03908785060048103), ('last', 0.0295556653290987), ('same', 0.021041089668869972)]
Top 5 next words: [('of', 0.463861882686615), ('when', 0.24241556227207184), ('or', 0.09381761401891708), ('then,', 0.04345019534230232), ('and', 0.03519059717655182)]
Top 5 next words: [('the', 0.8559945821762085), ('my', 0.03932814300060272), ('mr.', 0.026622071862220764), ('his', 0.018430126830935478), ('your', 0.014034280553460121)]
Top 5 next words: [('last', 0.5401196479797363), ('pool', 0.06053681671619415), ('door,', 0.018975136801600456), ('house,', 0.01414518989622593), ('<UNK>', 0.0134846530854702)]
Top 5 next words: [('london', 0.22870583832263947), ('few', 0.20850770175457), ('states', 0.09742055833339691), ('single', 0.07828395813703537), ('witness.', 0.03195361793041229)]
Top 5 next words: [('but', 0.5391944050788879), ('said', 0.1079741045832634), ('i.', 0.07954771816730499), ('for', 0.02788044884800911), ('as', 0.02153349481523037)]
Top 5 next words: [('the', 0.12480776011943817), ('i.', 0.11675642430782318), ('holmes', 0.09283915907144547), ('my', 0.08869172632694244), ('i', 0.05785762891173363)]
Top 5 next words: [('<EOS>', 0.9941174983978271), ('the', 0.0030809598974883556), ('my', 0.000953257898800075), ('your', 0.0003142689820379019), ('a', 0.00012030736252199858)]
Prompt: 'Watson was'
Generated: 'watson was at the moment of the last already,' but i.'
----------------------------------------
Top 5 next words: [('and', 0.15466643869876862), ('of', 0.1339583843946457), ('which', 0.06865507364273071), ('about', 0.04625558853149414), ('were', 0.03240438923239708)]
Top 5 next words: [('during', 0.13745912909507751), ('just', 0.1156328096985817), ('and', 0.1106153205037117), ('but', 0.09543406218290329), ('of', 0.0400993674993515)]
Top 5 next words: [('and', 0.4501616656780243), ('as', 0.17216673493385315), ('however,', 0.09286846965551376), ('came', 0.04033590108156204), ('but', 0.038192905485630035)]
Top 5 next words: [('i', 0.17947418987751007), ('the', 0.14824120700359344), ('we', 0.1295718252658844), ('my', 0.07865646481513977), ('a', 0.06920323520898819)]
Top 5 next words: [('have', 0.35155144333839417), ('may', 0.14480505883693695), ('had', 0.09444618970155716), ('came', 0.08784738928079605), ('could', 0.07874701917171478)]
Top 5 next words: [('been', 0.5048691034317017), ('heard', 0.31240150332450867), ('made', 0.031214535236358643), ('heard,', 0.02805810049176216), ('already', 0.012578816153109074)]
Top 5 next words: [('that', 0.4356001019477844), ('my', 0.11887266486883163), ('of', 0.09298993647098541), ('some', 0.06741493195295334), ('the', 0.06338461488485336)]
Top 5 next words: [('sound', 0.10234363377094269), ('man', 0.08601086586713791), ('little', 0.07794704288244247), ('young', 0.07432398200035095), ('only', 0.06572292745113373)]
Top 5 next words: [('of', 0.9993659853935242), ('and', 0.0002653866831678897), ('which', 0.00013683154247701168), ('in', 0.00012962859182152897), ('man', 3.366915916558355e-05)]
Top 5 next words: [('man', 0.3339461386203766), ('events,', 0.22399993240833282), ('of', 0.02377653308212757), ('<UNK>', 0.022678643465042114), ('our', 0.021177398040890694)]
Top 5 next words: [('which', 0.4415261447429657), ('and', 0.402180552482605), ('the', 0.02942156046628952), ('our', 0.027833785861730576), ('with', 0.01310028973966837)]
Top 5 next words: [('are', 0.1959277242422104), ('was', 0.1335587352514267), ('were', 0.12390213459730148), ('is', 0.09728418290615082), ('had', 0.09702048450708389)]
Top 5 next words: [('be', 0.4519069194793701), ('have', 0.19355995953083038), ('do', 0.042333681136369705), ('speak', 0.035690031945705414), ('cover', 0.017350593581795692)]
Top 5 next words: [('to', 0.5314347147941589), ('in', 0.1042737141251564), ('into', 0.07985417544841766), ('from', 0.07463087141513824), ('upon', 0.05639507994055748)]
Top 5 next words: [('the', 0.6482970118522644), ('his', 0.15429265797138214), ('your', 0.07224328070878983), ('you.', 0.051639024168252945), ('my', 0.010872597806155682)]
Top 5 next words: [('advertisement', 0.1830718219280243), ('general', 0.09077607840299606), ('matter.', 0.08853551745414734), ('official', 0.06700880825519562), ('<UNK>', 0.03810931369662285)]
Top 5 next words: [('and', 0.7410340905189514), ('from', 0.056823957711458206), ('sheet', 0.03276922181248665), ('which', 0.024427657946944237), ('with', 0.021571563556790352)]
Top 5 next words: [('of', 0.13930103182792664), ('which', 0.10824467986822128), ('to', 0.08732911944389343), ('in', 0.057747453451156616), ('from', 0.053705330938100815)]
Prompt: 'The detective'
Generated: 'the detective subduing just as i have heard the love of man which would speak to the advertisement and paper'
----------------------------------------
Top 5 next words: [('that', 0.5647698044776917), ('my', 0.06084911525249481), ('her', 0.055717166513204575), ('the', 0.05426477640867233), ('it', 0.04079709202051163)]
Top 5 next words: [('it', 0.31695473194122314), ('i', 0.25976434350013733), ('he', 0.09226618707180023), ('the', 0.07812374085187912), ('she', 0.06564026325941086)]
Top 5 next words: [('was', 0.754633903503418), ('would', 0.14120124280452728), ('had', 0.061569854617118835), ('is', 0.01591992750763893), ('could', 0.010669583454728127)]
Top 5 next words: [('not', 0.2814643383026123), ('a', 0.2515329122543335), ('no', 0.09011754393577576), ('nothing', 0.04878195747733116), ('an', 0.0448392853140831)]
Top 5 next words: [('one', 0.2333536446094513), ('doubt', 0.12279751896858215), ('great', 0.11470913141965866), ('sign', 0.06362618505954742), ('wonder', 0.04464488849043846)]
Top 5 next words: [('of', 0.4146765470504761), ('pass', 0.09104754030704498), ('to', 0.08847516775131226), ('about', 0.08743906021118164), ('which', 0.07698241621255875)]
Top 5 next words: [('to', 0.2827402949333191), ('for', 0.2044687122106552), ('is', 0.09733148664236069), ('in', 0.07170431315898895), ('came', 0.0670996680855751)]
Top 5 next words: [('get', 0.35676372051239014), ('go', 0.11304490268230438), ('make', 0.1037227138876915), ('tell', 0.08146451413631439), ('find', 0.0733337551355362)]
Top 5 next words: [('the', 0.3542419373989105), ('your', 0.3296046257019043), ('it', 0.057715240865945816), ('any', 0.057624220848083496), ('my', 0.04694584012031555)]
Top 5 next words: [('<UNK>', 0.17107148468494415), ('matter', 0.13178420066833496), ('man', 0.06602466851472855), ('case', 0.05144929885864258), ('easy', 0.05045687407255173)]
Top 5 next words: [('up,', 0.599224328994751), ('of', 0.06435661017894745), ('in', 0.05586996302008629), ('to', 0.04728386923670769), ('<UNK>', 0.046214886009693146)]
Top 5 next words: [('said', 0.5078865885734558), ('<UNK>', 0.06149223446846008), ('but', 0.05340610072016716), ('"of', 0.030518395826220512), ('at', 0.02939550019800663)]
Top 5 next words: [('he,', 0.9009114503860474), ('he.', 0.030760303139686584), ('i,', 0.027820590883493423), ('i.', 0.014393763616681099), ('i;', 0.007851573638617992)]
Top 5 next words: [("'and", 0.16060025990009308), ('"of', 0.1309751570224762), ('"but', 0.11069761961698532), ('smiling,', 0.10203079134225845), ('turning', 0.06453414261341095)]
Top 5 next words: [('me,', 0.26354897022247314), ('into', 0.15451517701148987), ('open', 0.09770163148641586), ('to', 0.08828584849834442), ('me', 0.051548171788454056)]
Top 5 next words: [('that', 0.14746078848838806), ('mr.', 0.14279575645923615), ('and', 0.1311279535293579), ('until', 0.06747191399335861), ('dark', 0.0477406270802021)]
Top 5 next words: [('and', 0.30031687021255493), ('to', 0.2414587140083313), ('in', 0.22068041563034058), ('as', 0.06032709777355194), ('or', 0.03511640802025795)]
Top 5 next words: [('eyes,', 0.11159378290176392), ('so', 0.09509867429733276), ('happened', 0.09263879805803299), ('servant', 0.07431317120790482), ('dropped', 0.03377898782491684)]
Prompt: 'I saw'
Generated: 'i saw that it is no one else to make the matter up," said he, turning me, dark and eyes,'
----------------------------------------
Top 5 next words: [('that', 0.4273039400577545), ('glance', 0.08492828905582428), ('more', 0.08106835186481476), ('faced', 0.043214112520217896), ('room,', 0.03990040719509125)]
Top 5 next words: [('that', 0.7264592051506042), ('i', 0.15431906282901764), ('room,', 0.017399631440639496), ('little', 0.010366713628172874), ('vague', 0.006679899524897337)]
Top 5 next words: [('i', 0.2595432996749878), ('he', 0.12264027446508408), ('she', 0.08638273924589157), ('my', 0.04873884841799736), ('thought', 0.04189915210008621)]
Top 5 next words: [('when', 0.12350807338953018), ('was', 0.11508574336767197), ('in', 0.041366394609212875), ('left', 0.037642695009708405), ('had', 0.03551981598138809)]
Top 5 next words: [('however,', 0.3512680232524872), ('but', 0.19248853623867035), ('<EOS>', 0.1711045801639557), ('at', 0.07272770255804062), ('i', 0.037161972373723984)]
Top 5 next words: [('the', 0.5099672079086304), ('last,', 0.06991942971944809), ('a', 0.0454423725605011), ('see', 0.04003835842013359), ('only', 0.03811102360486984)]
Top 5 next words: [("o'clock", 0.3879232108592987), ('minutes', 0.2242884486913681), ('same', 0.059238579124212265), ('time', 0.035461533814668655), ('i', 0.018669921904802322)]
Top 5 next words: [('i', 0.3528241813182831), ('she', 0.19774888455867767), ('he', 0.18308746814727783), ('we', 0.05490503087639809), ('but', 0.051822956651449203)]
Top 5 next words: [('man', 0.34460315108299255), ('she', 0.07769852131605148), ('i', 0.04614279791712761), ('cheetah', 0.03304464370012283), ('door', 0.032779548317193985)]
Top 5 next words: [('had', 0.5627781748771667), ('has', 0.09319290518760681), ('is', 0.07241788506507874), ('was', 0.06189895421266556), ('endeavoured', 0.04779623821377754)]
Top 5 next words: [('been', 0.2668614685535431), ('now', 0.12724153697490692), ('just', 0.12455647438764572), ('had', 0.06490432471036911), ('made', 0.04399166256189346)]
Top 5 next words: [('as', 0.2085222452878952), ('been', 0.17991039156913757), ('that', 0.12364010512828827), ('now', 0.06260184943675995), ('one', 0.05030852556228638)]
Top 5 next words: [('as', 0.2791258692741394), ('one.', 0.09961295127868652), ('in', 0.06425266712903976), ('<UNK>', 0.044653620570898056), ('to', 0.04148653894662857)]
Top 5 next words: [('<EOS>', 0.9999008178710938), ('i', 2.393215982010588e-05), ('it', 1.2523414625320584e-05), ('a', 7.579886187158991e-06), ('there', 2.506245664335438e-06)]
Prompt: 'It was a dark'
Generated: 'it was a dark so that railway marriage at ten o'clock a noise had just been one.'
----------------------------------------

```

## Key Findings

1. **Corpus Size Impact**: Small dataset (102K words) severely limits model performance
2. **Regularization Importance**: Dropout and layer normalization essential for small datasets
3. **Attention Effectiveness**: Additive attention improves context understanding
4. **Framework Comparison**: PyTorch offers better control over model architecture. I am aware of both architectures. When the same code was translated from pytorch to tensorflow, I saw performance improvement which seemed pretty strange.
5. **Overfitting Challenge**: Small corpus requires careful regularization to prevent memorization


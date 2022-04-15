# Import libraries
import os

# Math libraries to process the data
import nltk
import numpy as np
import pandas as pd

# Libraries for preprocessing
from nltk.tokenize import word_tokenize
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Classification libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Flatten,
    SpatialDropout1D,
    SimpleRNN,
    Dropout,
    LSTM,
    MaxPooling1D,
    GlobalMaxPool1D,
    Bidirectional,
    BatchNormalization,
)
from azureml.core import Experiment, Run, Workspace, Dataset
import azureml.core
nltk.download('punkt')

pd.options.mode.chained_assignment = None

print("Tensorflow version:", tensorflow.__version__)
print("Using GPU build:", tensorflow.test.is_built_with_cuda())
print("Is GPU available:", tensorflow.test.is_gpu_available())
print("SDK version:", azureml.core.VERSION)

outpus_folder = "./outputs"
os.makedirs(outpus_folder, exist_ok=True)

run = Run.get_context()

# Connect to the workspace
ws = run.experiment.workspace

dataframe_sample = Dataset.get_by_name(ws, name="Sample dataframe")
dataframe_sample = dataframe_sample.to_pandas_dataframe()

# Normalize target
dataframe_sample['target'] = dataframe_sample['target'] / 4

# Transform target from float to integer
dataframe_sample['target'] = dataframe_sample['target'].apply(lambda x: int(x))

# Vectorize (convert words to numbers) with Keras tokenizer
tk = Tokenizer(num_words=None)
tk.fit_on_texts(dataframe_sample.text)


def embed(corpus):
    return word_tokenizer.texts_to_sequences(corpus)


# Set text values
text_values = dataframe_sample.text.values

# Get longest sentence length
longest_train = max(text_values, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

# Vectorize (convert words to numbers) with Keras tokenizer
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(text_values)

# Get the dictionary of vocab created by Keras
word_index = tk.word_index

# Transform in fixed length structure
padded_sentences = pad_sequences(embed(text_values), length_long_sentence, padding='post')

# Set target values
sentiments = dataframe_sample.target.values

# Split the data in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    padded_sentences,
    sentiments,
    test_size=0.3
)

# Define path variable
glove_path = 'glove.6B.50d'

# Load GloVe vectors in a dictionary
embeddings_index = {}

glove = Dataset.get_by_name(ws, name=glove_path)
glove.to_pandas_dataframe()
for line in glove.to_pandas_dataframe()[['Line']].iteritems():
    values = line[1][0].split(' ')
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = vector

# Number of dimension of GloVe word embedding
GLOVE_DIM = 50

# Create embedding matrix for word in train set
embeddings_matrix_glove = np.zeros((len(word_index) + 1, GLOVE_DIM))
hits = 0
misses = 0

for word, i in tqdm(word_index.items()):
    # Check if the word occurs in Glove embedding
    embeddings_vector = embeddings_index.get(word)
    if embeddings_vector is not None:
        # If not, keep the vector with zeros only
        embeddings_matrix_glove[i] = embeddings_vector
        hits += 1
    else:
        misses += 1

print('\n')
print('Word index length: ', len(word_index) + 1)
print('Converted words: {}, missing words: {}'.format(hits, misses))
print('% of missing words: {:.1f}%'.format(misses / (hits + misses) * 100))

# Define model constants
epochs = 50
batch_size = 200

# Define the model
model = Sequential(name='glove_model')
model.add(Embedding(input_dim=embeddings_matrix_glove.shape[0],
                    output_dim=embeddings_matrix_glove.shape[1],
                    weights=[embeddings_matrix_glove],
                    input_length=length_long_sentence))
model.add(Bidirectional(LSTM(length_long_sentence, return_sequences=True, recurrent_dropout=0.2)))
model.add(GlobalMaxPool1D())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(length_long_sentence, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(length_long_sentence, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Take a look at the model summary
model.summary()

# Define model metrics
RECALL = tensorflow.keras.metrics.Recall(name='recall')
PRECISION = tensorflow.keras.metrics.Precision(name='precision')
METRICS = [RECALL, PRECISION]

# Compile the model
model.compile(loss='binary_crossentropy', optimizer="adam", metrics=METRICS)

# Fit the model
hist = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.3,
)

run.log_list("Training Recall", hist.history["recall"])
run.log_list("Validation Recall", hist.history["val_recall"])
run.log_list("Training Precision", hist.history["precision"])
run.log_list("Validation Precision", hist.history["val_precision"])

# Evaluate the model
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Test recall:", score[1])
print("Test precision:", score[2])
print("recall", score[0])
print("precision", score[1])

project_folder = "./sentiment_analysis_logtoazure"
os.makedirs(project_folder, exist_ok=True)

keras_path = os.path.join(project_folder, "keras")
os.makedirs(keras_path, exist_ok=True)

with open(os.path.join(keras_path, "model.json"), "w") as f:
    f.write(model.to_json())
model.save_weights(os.path.join(keras_path, "model.pkl"))

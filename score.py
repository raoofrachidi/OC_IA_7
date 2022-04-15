import json
import numpy as np
from azureml.core.model import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import logging
import pickle
from time import time
import os


# Called when the service is loaded
def init():
    global glove_model
    global tokenizer

    logging.basicConfig(level=logging.DEBUG)
    # Get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sentiment_model')

    # Load existing model
    glove_model = load_model(model_path + '/sentiment_model.h5')

    # Load tokenizer
    with open(model_path + '/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)


# Handle request to the service
def run(data):
    try:
        # Pick out the text property of the JSON request
        # Expected JSON details {"text": "some text to score for sentiment"}
        data = json.loads(data)
        prediction = predict(data['text'])
        return prediction
    except Exception as e:
        error = str(e)
        return error


# Determine sentiment from score
NEGATIVE = 'NEGATIVE'
POSITIVE = 'POSITIVE'


def decode_sentiment(score):
    return NEGATIVE if score < 0.5 else POSITIVE


# Predict sentiment using the model
SEQUENCE_LENGTH = 46


def predict(text):
    start = time()

    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]),
                           maxlen=SEQUENCE_LENGTH)

    # Predict
    score = glove_model.predict([x_test])[0]

    # Decode sentiment
    label = decode_sentiment(score)

    return {'label': label, 'score': float(score),
            'elapsed_time': time() - start}

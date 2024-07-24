#!/usr/bin/env python3

from flask import Flask
from tensorflow.keras.models import load_model
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences

loaded_model = load_model('model.h5')

def preprocess_input(text, tokenizer, max_sequence_length):
	tokenized_text = tokenizer.texts_to_sequences([text])
	padded_text = pad_sequences(tokenized_text, maxlen=max_sequence_length, padding='post')
	return padded_text

def logits_to_text(logits, tokenizer):
	index_to_words = {id: word for word, id in tokenizer.word_index.items()}
	index_to_words[0] = ''
	return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def final_predictions(model, input_text, english_tokenizer, french_tokenizer, max_french_sequence_length):
	preprocessed_text = preprocess_input(input_text, english_tokenizer, max_french_sequence_length)
	prediction = model.predict(preprocessed_text, verbose=0)[0]
	translated_text = logits_to_text(prediction, french_tokenizer)
	return translated_text

while True:
	try:
		txt = input("Enter English text for translation (Ctrl+C to exit): ").lower()
		translation = final_predictions(loaded_model, txt, english_tokenizer, french_tokenizer, preproc_french_sentences.shape[1])
		print("French Translation:", translation)
	except KeyboardInterrupt:
		print("\nExiting...")
		break
	except:
		print("Word not recognized. Please try again.")

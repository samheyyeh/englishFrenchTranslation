#!/usr/bin/env python3

import helper
import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Dropout, LSTM
from keras.layers import Embedding 
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import os
import collections
import re
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

english_path='https://raw.githubusercontent.com/projjal1/datasets/master/small_vocab_en.txt'
french_path='https://raw.githubusercontent.com/projjal1/datasets/master/small_vocab_fr.txt'

def load_data(path):
  input_file = os.path.join(path)
  with open(input_file, "r") as f:
    data = f.read()

  return data.split('\n')

english_data=tf.keras.utils.get_file('file1',english_path)
french_data=tf.keras.utils.get_file('file2',french_path)    

english_sentences=load_data(english_data)
french_sentences=load_data(french_data)

for i in range(5):
  print('Sample :',i)
  print(english_sentences[i])
  print(french_sentences[i])
  print('-'*50)    

english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

print('English Vocab:',len(english_words_counter))
print('French Vocab:',len(french_words_counter))

def tokenize(x):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(x)
  return tokenizer.texts_to_sequences(x), tokenizer

text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']

text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()

for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
  print('Sequence {} in x'.format(sample_i + 1))
  print('  Input:  {}'.format(sent))
  print('  Output: {}'.format(token_sent))

def pad(x, length=None):
  return pad_sequences(x, maxlen=length, padding='post')    

def preprocess(x, y):
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

def logits_to_text(logits, tokenizer):
  index_to_words = {id: word for word, id in tokenizer.word_index.items()}
  index_to_words[0] = ''

  return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 0.005
    
    model = Sequential()
    model.add(Embedding(english_vocab_size, 256, input_length=input_shape[1], input_shape=input_shape[1:]))
    model.add(GRU(256, return_sequences=True))    
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 

    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
     
tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))

simple_rnn_model = embed_model(
    tmp_x.shape,
    preproc_french_sentences.shape[1],
    len(english_tokenizer.word_index)+1,
    len(french_tokenizer.word_index)+1)
     
simple_rnn_model.summary()

history=simple_rnn_model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.2)
     
simple_rnn_model.save('model.h5')

def final_predictions(text):
  y_id_to_word = {value: key for key, value in french_tokenizer.word_index.items()}
  y_id_to_word[0] = ''

  sentence = [english_tokenizer.word_index[word] for word in text.split()]
  sentence = pad_sequences([sentence], maxlen=preproc_french_sentences.shape[-2], padding='post')
  
  print(sentence.shape)
  print(logits_to_text(simple_rnn_model.predict(sentence[:1])[0], french_tokenizer))

print("Finished training model!")

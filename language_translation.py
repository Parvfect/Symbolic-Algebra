
# https://analyticsindiamag.com/sequence-to-sequence-modeling-using-lstm-for-language-translation/

import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.utils import *
from keras.initializers import *
import tensorflow as tf
import time, random

file_name = "fra.txt"

# Hyperparameters
batch_size = 64
latent_dim = 256
num_samples = 10000

# Vectorizing data
input_texts = []
target_texts = []
input_chars = set()
target_chars = set()

def encode_data(file_namem, input_chars, target_chars, num_samples, input_texts, target_texts):
    with open(f'{file_name}', 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')

    for line in lines[1: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_chars:
                input_chars.add(char)
        for char in target_text:
            if char not in target_chars:
                target_chars.add(char)

    input_chars = sorted(list(input_chars))
    target_chars = sorted(list(target_chars))
    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    #Print size
    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    return input_chars, target_chars, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length

def define_encoder_decoder_data(input_chars, target_chars, num_encoder_tokens, num_decoder_tokens, max_encoder_seq_length, max_decoder_seq_length):

    # Define data for encoder and decoder
    input_token_id = dict([(char, i) for i, char in enumerate(input_chars)])
    taget_token_id = dict([(char, i) for i, char in enumerate(target_chars)])

    encoder_in_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')

    decoder_in_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_in_data[i, t, input_token_id[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_in_data[i, t, target_token_id[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_id[char]] = 1.

    return encoder_in_data, decoder_in_data, decoder_target_data

def process_input_sequence(num_encoder_tokens, num_decoder_tokens):
    
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Using encoder states to set up the deecoder as initial state
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    return encoder_inputs, decoder_inputs, decoder_outputs, encoder_states, decoder_lstm, decoder_dense

def get_model_characteristics(encoder_inputs, decoder_inputs, decoder_outputs, encoder_in_data, decoder_in_data, decoder_target_data):

    # Final Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    #Model data Shape
    print("encoder_in_data shape:",encoder_in_data.shape)
    print("decoder_in_data shape:",decoder_in_data.shape)
    print("decoder_target_data shape:",decoder_target_data.shape)
    
    #Visualize the model
    plot_model(model,show_shapes=True)

def train_model():
    model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2 = 0.999, decay=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit([encoder_in_data, decoder_in_data], decoder_target_data, batch_size=batch_size, epochs=50, validation_split=0.2)

def define_inference_models():
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    reverse_input_char_index = dict((i, char) for char, i in input_token_id.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_id.items())

    def decoding_sequence(input_seq):
        states_value = encoder_model.predict(input_seq)

        #Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        #Get the first character of target sequence with the start character.
        target_seq[0, 0, target_token_id['\t']] = 1.

        #Sampling loop for a batch of sequences
        #(to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            #Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            #Exit condition: either hit max length
            #or find stop character.
            if (sampled_char == '\n' or
            len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            #Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            #Update states
            states_value = [h, c]

        return decoded_sentence

    for seq_index in range(10):
        input_seq = encoder_in_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

def main():
    input_texts, target_texts, input_chars, target_chars = encode_data(file_name, input_chars, target_chars, input_texts, target_texts)
    encoder_in_data, decoder_in_data, decoder_target_data = define_encoder_decoder_data(input_texts, target_texts, input_chars, target_chars)
    encoder_inputs, decoder_inputs, decoder_outputs, encoder_states, decoder_lstm, decoder_dense = process_input_sequence(num_encoder_tokens, num_decoder_tokens)
    get_model_characteristics(encoder_inputs, decoder_inputs, decoder_outputs, encoder_in_data, decoder_in_data, decoder_target_data)


main()
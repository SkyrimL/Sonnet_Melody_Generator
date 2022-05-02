#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse
import midi_statistics
import numpy as np
import os
import pandas as pd
import pyphen
import random
import string
import tensorflow as tf
import utils

from gensim.models import Word2Vec
from helpers import *
from model import *

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    line = 1

    # read words with rhyme
    rhyme = pd.read_pickle("word_to_rhymes.pkl")
    prev_word = None

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]

        # Generating rhyme
        if "\n" in predicted_char:
            if line % 2 == 0:
                words = predicted.rsplit(" ", 1)
                prev = words[0]
                word = words[1]
                punc = ""
                if word[-1] in string.punctuation:
                    punc = word[-1]
                    word = word[:-1]

                if not prev_word:
                    if word not in rhyme:
                        word = random.choice(list(rhyme.keys()))
                else:
                    candidate = set(rhyme[prev_word]).intersection(list(rhyme.keys()))
                    if candidate == set():
                        word = random.choice(list(rhyme.keys()))
                    else:
                        word = random.choice(list(candidate))
                   
                predicted = " ".join([prev, word + punc]) 
                prev_word = word
                #print(word, word in rhyme)
            line += 1

        predicted += predicted_char

        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted


def style_cleaner(sonnet):
    lines = sonnet.split("\n")
    new_sonnet = "\n".join(lines[:14])
    return new_sonnet


def remove_punc(sonnet):
    lines = sonnet.split("\n")
    lines = [l.strip() for l in lines]
    lines = [l.strip(string.punctuation) for l in lines]
    return "\n".join(lines[:14])


def generate_pairs(sonnet):
    dic = pyphen.Pyphen(lang='en')

    line_pairs = []
    lines = sonnet.split("\n")
    for line in lines:
        pairs = []
        words = lines.split()
        for word in words:
            word = word.strip(string.punctuation)
            syllables = dic.inserted(word)
            syllables = syllables.strip("-").split("-")
            for syl in syllables:
                pairs.append([syl, word])
        line_pairs.append(pairs)
    
    return np.array(line_pairs)

    


# Run as standalone script
if __name__ == '__main__':

    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--filename', type=str, default="sonnets2.pt")
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=1000)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    del args.filename

    # generate sonnet
    sonnet = generate(decoder, **vars(args))
    sonnet = style_cleaner(sonnet)

    # remove punctuations
    cleaned_sonnet = remove_punc(sonnet)
    print(cleaned_sonnet)

    
    # load word to vector models
    model_path = './saved_gan_models/saved_model_best_overall_mmd'
    syll_model_path = './enc_models/syllEncoding_20190419.bin'
    word_model_path = './enc_models/wordLevelEncoder_20190419.bin'
    syllModel = Word2Vec.load(syll_model_path)
    wordModel = Word2Vec.load(word_model_path)

    # generate syllables and word pairs
    dic = pyphen.Pyphen(lang='en')
    entire_lyrics = []
    lines = sonnet.split("\n")
    for line in lines:
        temp_pairs = []
        words = line.split()
        for word in words:
            word = word.strip(string.punctuation)
            syllables = dic.inserted(word)
            syllables = syllables.strip("-").split("-")
            for syl in syllables:
                temp_pairs.append([syl, word])

        # clean pairs
        pairs = []
        for syll, word in temp_pairs:
            if syll in syllModel.wv.key_to_index and word in wordModel.wv.key_to_index:
                pairs.append([syll, word])
        entire_lyrics.append(pairs)
    
    # generate melody in segments
    index = 1
    for lyrics in entire_lyrics:
        length_song = len(lyrics)
        cond = []

        for i in range(20):
            if i < length_song:
                syll2Vec = syllModel.wv[lyrics[i][0]]
                word2Vec = wordModel.wv[lyrics[i][1]]
                cond.append(np.concatenate((syll2Vec,word2Vec)))
            else:
                cond.append(np.concatenate((syll2Vec,word2Vec)))


        flattened_cond = []
        for x in cond:
            for y in x:
                flattened_cond.append(y)

        x_list = []
        y_list = []

        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, [], model_path)
            graph = tf.get_default_graph()
            keep_prob = graph.get_tensor_by_name("model/keep_prob:0")
            input_metadata = graph.get_tensor_by_name("model/input_metadata:0")
            input_songdata = graph.get_tensor_by_name("model/input_data:0")
            output_midi = graph.get_tensor_by_name("output_midi:0")
            feed_dict = {}
            feed_dict[keep_prob.name] = 1.0
            condition = []
            feed_dict[input_metadata.name] = condition
            feed_dict[input_songdata.name] = np.random.uniform(size=(1, 20, 3))
            condition.append(np.split(np.asarray(flattened_cond), 20))
            feed_dict[input_metadata.name] = condition
            generated_features = sess.run(output_midi, feed_dict)
            sample = [x[0, :] for x in generated_features]
            sample = midi_statistics.tune_song(utils.discretize(sample))
            midi_pattern = utils.create_midi_pattern_from_discretized_data(sample[0:length_song])
            destination = "melody{0}.mid".format(index)
            midi_pattern.write(destination)
            
            print('done', index)
        
        index += 1
        
    print("END")


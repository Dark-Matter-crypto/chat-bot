import os
import numpy
import nltk
import random
import json
# import tensorflow
# import tflearn
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from django.shortcuts import render, HttpResponse


def load_data_file():
    module_dir = os.path.dirname(__file__)
    intents_file_path = os.path.join(module_dir, 'static\index\data\intents.json')
    with open(intents_file_path) as intents_file:
        data = json.load(intents_file)

    return data


# Create your views here.

def index_view(request):
    data = load_data_file()

    words = []
    tags = []
    patterns = []
    pattern_tags = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern_words = nltk.word_tokenize(pattern)
            words.extend(pattern_words)
            patterns.append(pattern_words)
            pattern_tags.append(intent['tag'])

        if intent['tag'] not in tags:
            tags.append(intent['tag'])

    words = [stemmer.stem(word.lower()) for word in words if w != "?"]
    words = sorted(list(set(words)))

    tags = sorted(tags)

    training = []
    output = []

    out_empty = [0 for _ in range(len(tags))]

    for p, pattern in enumerate(patterns):
        bag = []
        pattern_words = [stemmer.stem(word) for word in pattern]

        for word in words:
            if word in pattern_words:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[tags.index(pattern_tags[p])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)





    return HttpResponse(tags)

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


# Create your views here.


def index_view(request):
    module_dir = os.path.dirname(__file__)
    intents_file_path = os.path.join(module_dir, 'static\index\data\intents.json')
    with open(intents_file_path) as intents_file:
        data = json.load(intents_file)

    words = []
    tags = []
    patterns = []
    pattern_tags = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            #pattern_words = nltk.word_tokenize(pattern)
            #words.extend(pattern_words)
            patterns.append(pattern)
            pattern_tags.append(intent['tag'])

        if intent['tag'] not in tags:
            tags.append(intent['tag'])

    words = [stemmer.stem(word.lower()) for word in words]
    words = sorted(list(set(words)))

    tags = sorted(tags)


    return HttpResponse(tags)

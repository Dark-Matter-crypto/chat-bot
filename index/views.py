import os
import numpy
import nltk
import random
import json
import pickle
# import tensorflow
# import tflearn
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from django.shortcuts import render, HttpResponse


def load_intents_file(module_dir):
    intents_file_path = os.path.join(module_dir, 'static\index\data\intents.json')
    with open(intents_file_path) as intents_file:
        data = json.load(intents_file)

    return data


def save_data_file(module_dir, words, tags, training, output):
    data_file_path = os.path.join(module_dir, 'static\index\data\data.pickle')
    with open(data_file_path, "wb") as data_file:
        pickle.dump((words, tags, training, output), data_file)


def save_model(module_dir, model):
    model_path = os.path.join(module_dir, 'static\index\model\model.tflearn')
    model.save(model_path)


def open_model(module_dir, model):
    model_path = os.path.join(module_dir, 'static\index\model\model.tflearn')
    model.load(model_path)
    return model


def setup_model(training, output):
    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)


# Create your views here.

def index_view(request):
    module_dir = os.path.dirname(__file__)

    try:
        data_file_path = os.path.join(module_dir, 'static\index\data\data.pickle')
        with open(data_file_path, "rb") as data_file:
            words, tags, training, output = pickle.load(data_file)

    except FileNotFoundError:
        data = load_intents_file(module_dir)
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

        save_data_file(module_dir, words, tags, training, output)

    model = setup_model(training, output)

    try:
        open_model(module_dir, model)
    except FileNotFoundError:
        model.fit(training, output, n_epoch=5000, batch_size=8, show_metric=True)
        save_model(module_dir, model)
        
    return HttpResponse(tags)

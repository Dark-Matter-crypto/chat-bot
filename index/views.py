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
    return model


def user_bag_of_words(user_input, words):
    bag = [0 for _ in range(len(words))]
    user_words = nltk.word_tokenize(user_input)
    user_words = [stemmer.stem(word.lower()) for word in user_words]

    for word in user_words:
        for i, w in enumerate(user_input):
            if w == word:
                bag[i] = 1

    return numpy.array(bag)


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

    model_path = os.path.join(module_dir, 'static\index\model\model.tflearn')

    try:
        model.load(model_path)
    except:
        model.fit(training, output, n_epoch=5000, batch_size=8, show_metric=True)
        model.save(model_path)

    unknown_query_response = [
        "Hmmm...Not sure what you asking there.",
        "This is me telling you I didn't understand what you just said.",
        "Not sure what you asking there. Sorry.",
        "Didn't catch that. Please try another question.",
        "Sorry, didn't understand your question.",
        "Escuze moire?"
    ]
    results = model.predict([user_bag_of_words(user_input, words)])[0]
    results_index = numpy.argmax(results)
    tag = tags[results_index]

    if results[results_index] > 0.85:
        for intent in data['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']
        response = random.choice(responses)
    else:
        response = random.choice(unknown_query_response)

    return HttpResponse(response)

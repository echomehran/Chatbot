import tflearn
from tensorflow.python.framework import ops 
import random 
import json  
import numpy as np  
import nltk  
from nltk.stem.lancaster import LancasterStemmer
import pickle  

# nltk.download ('all')

import os  

stemmer = LancasterStemmer()


with open('') as file:
    data = json.load(file)

try:
    with open('', '') as f:
        words, labels, training, output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)

            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in '?']
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open('', '') as f:
        pickle.dump((words, labels, training, output), f)

try:
    model.load('model.tflearn')
except:
    ops.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=2500, batch_size=8, show_metric=True)
    model.save('model.tflearn')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_wordds = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_wordds]

    for se in s_wordds:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    print('\nStart talking with the bot (type q or quit to stop)!')

    while True:
        inp = input('You: ')
        if inp.lower() == 'q' or inp.lower() == 'quit':
            print('\nGoodbye')
            break
        if inp.lower() == 'clear' or inp.lower() == 'cls':
            os.system('clear')


        res = model.predict([bag_of_words(inp, words)])[0]
        print(res)
        res_index = np.argmax(res)
        print(res_index)
        tag = labels[res_index]
        print(res[res_index])

        if res[res_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I didn't get that, try again.")

if __name__ == '__main__':
    try:
        chat()
    except KeyboardInterrupt:
        print('\nGoodbye')

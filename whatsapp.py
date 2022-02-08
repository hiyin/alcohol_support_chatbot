import os
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
#
# import subprocess
# subprocess.run(['python', 'chatbot.py'])

from keras.models import load_model
model = load_model('Chatbot_model.h5')
import json
import random
intents = json.loads(open('intents_cn.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model, context="{'123':['問候']}", userID='123'):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    #print(res)
    ERROR_THRESHOLD = 0.01
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        #print(classes[r[0]])
        if context[userID] == ['问候'] and context[userID] == classes[r[0]]:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        else:
            #print('yesyes')
            if classes[r[0]] in context[userID]:
                return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# New response function (contextual)
def getResponse(sentence, context, userID='123'):
    results = predict_class(sentence, model, context, userID='123')
    # print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # print(i)
                # print(i['tag'])
                # print(results[0]['intent'])
                # find a tag matching the first result
                if not context:
                    if i['tag'] == results[0]['intent']:
                        print("matching ...")
                        # set context for this intent if necessary
                        if 'context' in i:
                            context[userID] = i['context']
                            return random.choice(i['responses'])
                else:
                    if i['tag'] == results[0]['intent'] and i['tag'] in context[userID]:
                        print("matching ...")
                        # set context for this intent if necessary
                        if 'context' in i:
                            context[userID] = i['context']
                            return random.choice(i['responses'])

            results.pop(0)


def chatbot_response(text):
    ### Old response
    # ints = predict_class(text, model)
    # res = getResponse(ints, intents)
    ### New response (contextual)
    res = getResponse(text, context, userID='123')
    print(context)
    return res


app = Flask(__name__)
account = "AC45abd6b358532bbe609bdd4d57f83fc9"
token = "3e8fc5acf8f84fe48224689554f82fb9"
client = Client(account, token)


def respond(message):
    response = MessagingResponse()
    response.message(message)
    return Response(str(response), mimetype="application/xml")

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/message', methods=['POST'])
def reply():
    message = request.form.get('Body').lower()
    # greeting
    if message:
        reply = chatbot_response(message)
        return respond(reply)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
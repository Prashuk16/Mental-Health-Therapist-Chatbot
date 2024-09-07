import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

# Load the model and necessary files
model = load_model('C:\\Users\\prash\Mental-health-Chatbot\\new_model.h5')


words = pickle.load(open("C:\\Users\\prash\\Mental-health-Chatbot\\new_texts.pkl", 'rb'))

classes = pickle.load(open("C:\\Users\\prash\\Mental-health-Chatbot\\new_labels.pkl", 'rb'))


intents = json.loads(open('C:\\Users\\prash\\Mental-health-Chatbot\\intents.json').read())

lemmatizer = WordNetLemmatizer()


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)
# Predict the intent class based on the input sentence
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    print(res)
    ERROR_THRESHOLD = 0.0
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
# Get the appropriate response based on the predicted intent
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Function to handle the chatbot's response
def chatbot_response(msg):
    ints = predict_class(msg, model)
    if(len(ints)==0):
        return "Can't get you, please come again!"
    res = getResponse(ints, intents)
    return res

# Flask setup for the chatbot interface
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()
    
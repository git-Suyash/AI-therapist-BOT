import sqlite3
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import keras
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
import random

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

# Load the model, words, and classes
model = keras.models.load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Connect to the SQLite database
def get_db_connection():
    conn = sqlite3.connect('chatbot_database.db')
    conn.row_factory = sqlite3.Row
    return conn

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_product_details(product_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT description, price FROM products WHERE name LIKE ?', (f'%{product_name}%',))
    result = cursor.fetchone()
    conn.close()
    if result:
        description, price = result
        return f"The {product_name} is {description} and costs ${price:.2f}."
    else:
        return f"Sorry, I couldn't find any details for {product_name}."

def get_response(ints, intents_json, message):
    tag = ints[0]['intent'] if ints else None
    print(f"Detected intent: {tag}, Message: {message}")  # Debugging statement  
    list_of_intents = intents_json['intents']
    response = "Sorry, I couldn't understand your query."

    for i in list_of_intents:
        if i['tag'] == tag:
            print(f"Identified tag: {tag}")  # Debugging statement
            if tag == 'product_name':
                words_in_message = clean_up_sentence(message)
                product_name = None
                print(f"Words in message: {words_in_message}")  # Debugging statement
                for word in words_in_message:
                    cursor = get_db_connection().cursor()
                    cursor.execute('SELECT name FROM products WHERE name LIKE ?', (f'%{word}%',))
                    if cursor.fetchone():
                        product_name = word
                        break

                print(f"Extracted product name: {product_name}")  # Debugging statement

                if product_name:
                    response = get_product_details(product_name)
                else:
                    response = "Sorry, I couldn't identify the product name. Can you please specify?"
            else:
                response = random.choice(i['responses'])
            break
    return response

@app.route('/chat', methods=['POST'])
def chat():
    global message
    message = request.json['message']
    print(f"User message: {message}")  # Debugging statement
    ints = predict_class(message, model)
    print(f"Predicted intents: {ints}")  # Debugging statement
    res = get_response(ints, intents, message)
    print(f"Bot response: {res}")  # Debugging statement

    # Log user interaction to the database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO interactions (message, response) VALUES (?, ?)', (message, res))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging interaction: {e}")

    return jsonify({"response": res})

@app.route('/')
def index():
    return send_from_directory('', 'chatbot_index.html')

if __name__ == "__main__":
    app.run(debug=True)



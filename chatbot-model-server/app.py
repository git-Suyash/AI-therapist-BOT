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
    sentence_words = nltk.word_tokenize(sentence) #tokenize sentence into words.
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words] # Lemmatize each word to its base form and convert to lowercase.
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)  # Clean up the input sentence.
    bag = [0]*len(words)  # Initialize a bag of words with zeros.
    for s in sentence_words: #Create the bag of words array.
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1 # If the word is in the sentence, set the corresponding position in the bag to 1.
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)  # Convert the input sentence into a bag of words.
    res = model.predict(np.array([p]))[0]   # Predict the class probabilities for the input sentence.
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]  # Define a threshold to filter out low probability predictions and collect prediction above threshold.
    results.sort(key=lambda x: x[1], reverse=True) #sort result in descending order.
    return_list = []
    for r in results: # Prepare the final list of predictions with their corresponding intents and probabilities.
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#Conection to database for fetching products details.
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

#fetch the intents from json file and if not found give default message 
def get_response(ints, intents_json, message):
    tag = ints[0]['intent']  
    list_of_intents = intents_json['intents']
    response = "Sorry, I couldn't understand your query."

    for i in list_of_intents: # Iterate through the list of intents
        if i['tag'] == tag:
            if tag == 'product_name':   # Special handling for the 'product_name' intent
                words_in_message = clean_up_sentence(message)
                product_name = None
                for word in words_in_message: # Iterate through the words in the message
                    cursor = get_db_connection().cursor()
                    cursor.execute('SELECT name FROM products WHERE name LIKE ?', (f'%{word}%',))   # Execute a SQL query to find a product name matching the word
                    if cursor.fetchone():
                        product_name = word
                        break

                if product_name:
                    response = get_product_details(product_name)  # If a product name was identified, get the product details
                else:
                    response = "Sorry, I couldn't identify the product name. Can you please specify?"
            else:
                response = random.choice(i['responses'])    # For other intents, randomly select a response from the intent's responses
            break
    return response

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']  # Extract the message from the JSON request
    ints = predict_class(message, model)  # Predict the class of the message using the model
    res = get_response(ints, intents, message)

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

# Define the route for the home page
@app.route('/')
def index():
    return send_from_directory('', 'chatbot_index.html')   # Serve the 'chatbot_index.html' file from the current directory

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from fuzzywuzzy import process
import json
import os
import string

app = Flask(__name__)

# File to store the knowledge base
KNOWLEDGE_FILE = "knowledge_base.json"

def load_knowledge_base():
    """Load the knowledge base from a file."""
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_knowledge_base(knowledge_base):
    """Save the knowledge base to a file."""
    with open(KNOWLEDGE_FILE, "w") as file:
        json.dump(knowledge_base, file, indent=4)

def normalize_input(input_text):
    """Normalize the input by removing extra spaces, punctuation, and converting to lowercase."""
    input_text = input_text.translate(str.maketrans("", "", string.punctuation))
    input_text = " ".join(input_text.lower().split())
    return input_text

def get_response(knowledge_base, question):
    """Get a response from the knowledge base using fuzzy matching."""
    question = normalize_input(question)
    if not knowledge_base:  # Check if the knowledge base is empty
        return None
    matches = process.extract(question, knowledge_base.keys(), limit=1)
    if matches:  # Check if any matches were found
        closest_question, score = matches[0]
        if score >= 80:  # Adjust the threshold as needed
            return knowledge_base[closest_question]
    return None

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    """Handle chat bot requests."""
    user_input = request.json.get("input")
    knowledge_base = load_knowledge_base()
    response = get_response(knowledge_base, user_input)
    if response:
        return jsonify({"response": response})
    else:
        return jsonify({"response": "I don't know the answer to that. Can you teach me?"})

@app.route("/teach_bot", methods=["POST"])
def teach_bot():
    """Teach the bot a new response."""
    user_input = request.json.get("input")
    answer = request.json.get("answer")
    knowledge_base = load_knowledge_base()
    knowledge_base[normalize_input(user_input)] = answer
    save_knowledge_base(knowledge_base)
    return jsonify({"response": "Thank you! I've learned something new. 💖"})

if __name__ == "__main__":
    app.run(debug=True)
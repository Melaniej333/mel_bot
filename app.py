from flask import Flask, render_template, request, jsonify
from rapidfuzz import process
from sentence_transformers import SentenceTransformer, util
import json
import os
import string

app = Flask(__name__)

KNOWLEDGE_FILE = "knowledge_base.json"
model = SentenceTransformer('all-MiniLM-L6-v2')

class QuestionContext:
    def __init__(self):
        self.original_question = None
        self.possible_matches = []
        self.waiting_for_answer = False
        self.current_question = None
        self.waiting_for_correction = False
        self.question_to_correct = None

question_context = QuestionContext()

def load_knowledge_base():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_knowledge_base(knowledge_base):
    with open(KNOWLEDGE_FILE, "w") as file:
        json.dump(knowledge_base, file, indent=4)

def normalize_input(input_text):
    input_text = input_text.translate(str.maketrans("", "", string.punctuation))
    input_text = " ".join(input_text.lower().split())
    return input_text

def get_response(knowledge_base, question):
    normalized_question = normalize_input(question)
    question_context.current_question = normalized_question
    
    if not knowledge_base:
        return {
            "type": "learning",
            "response": "I don't know the answer to that. Can you teach me?"
        }

    matches = process.extract(normalized_question, knowledge_base.keys(), limit=1)
    fuzzy_score = 0
    fuzzy_response = None
    if matches:
        closest_question, fuzzy_score, _ = matches[0]
        fuzzy_response = knowledge_base[closest_question]

    question_embedding = model.encode(normalized_question, convert_to_tensor=True)
    knowledge_base_questions = list(knowledge_base.keys())
    knowledge_base_embeddings = model.encode(knowledge_base_questions, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, knowledge_base_embeddings)[0]
    best_match_index = similarities.argmax().item()
    best_match_score = similarities[best_match_index].item()
    best_match_question = knowledge_base_questions[best_match_index]
    semantic_response = knowledge_base[best_match_question]

    fuzzy_threshold = 80
    semantic_threshold = 0.7

    question_context.original_question = normalized_question

    # Direct match found with high confidence
    if fuzzy_score >= fuzzy_threshold or best_match_score >= semantic_threshold:
        best_match = closest_question if fuzzy_score >= fuzzy_threshold else best_match_question
        question_context.question_to_correct = best_match
        return {
            "type": "answer",
            "response": knowledge_base[best_match]
        }

    # No good match found
    return {
        "type": "learning",
        "response": "I don't know the answer to that. Can you teach me?"
    }

def update_answer(knowledge_base, question, new_answer):
    """Update the answer for a question and all its similar questions."""
    if question in knowledge_base:
        knowledge_base[question] = new_answer
        save_knowledge_base(knowledge_base)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_bot_response():
    try:
        user_input = request.json.get("input", "").strip()
        if not user_input:
            return jsonify({
                "type": "error",
                "response": "Please enter a message."
            })

        knowledge_base = load_knowledge_base()
        normalized_input = normalize_input(user_input)

        # Handle "no" response and corrections
        if question_context.question_to_correct and normalized_input == "no":
            question_context.waiting_for_correction = True
            return jsonify({
                "type": "correction",
                "response": "What is the correct answer?"
            })

        # Handle the correction if we're waiting for one
        if question_context.waiting_for_correction:
            update_answer(knowledge_base, question_context.question_to_correct, user_input)
            # Reset the correction context
            question_context.waiting_for_correction = False
            question_context.question_to_correct = None
            return jsonify({
                "type": "success",
                "response": "Oops thanks for the correction! I've updated my knowledge database! <3"
            })

        # If we're waiting for an answer to a new question
        if question_context.waiting_for_answer:
            knowledge_base[question_context.original_question] = user_input
            save_knowledge_base(knowledge_base)
            
            # Reset all context
            question_context.waiting_for_answer = False
            question_context.original_question = None
            question_context.possible_matches = []
            question_context.current_question = None
            
            return jsonify({
                "type": "success",
                "response": "Thank you! I've learned something new. 💖"
            })

        # Normal flow for new questions
        response = get_response(knowledge_base, user_input)
        return jsonify(response)

    except Exception as e:
        print(f"Error in get_bot_response: {str(e)}")
        return jsonify({
            "type": "error",
            "response": "I encountered an error. Please try again."
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Get Render's assigned port
    app.run(host="0.0.0.0", port=port)  # Bind to 0.0.0.0
    app.run(debug=True)
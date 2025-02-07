from flask import Flask, render_template, request, jsonify
from rapidfuzz import process
from sentence_transformers import SentenceTransformer, util
import json
import os
import string

class QuestionContext:
    def __init__(self):
        self.original_question = None
        self.possible_matches = []
        self.waiting_for_answer = False
        self.current_question = None
        self.waiting_for_correction = False
        self.question_to_correct = None
        self.suggested_questions = []
        self.last_question = None
        self.is_clarifying = False
        self.needs_correction = False

question_context = QuestionContext()

app = Flask(__name__)

KNOWLEDGE_FILE = "knowledge_base.json"
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_knowledge_base():
    if os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_knowledge_base(knowledge_base):
    with open(KNOWLEDGE_FILE, "w") as file:
        json.dump(knowledge_base, file, indent=4)
    print("Knowledge base saved successfully!")

def normalize_input(input_text):
    input_text = input_text.translate(str.maketrans("", "", string.punctuation))
    input_text = " ".join(input_text.lower().split())
    return input_text

def update_answer(knowledge_base, question, new_answer):
    """Update the answer for a question and all its similar questions."""
    print(f"Updating knowledge base with: {question} -> {new_answer}")
    knowledge_base[question] = new_answer
    save_knowledge_base(knowledge_base)

def store_similar_question(knowledge_base, original_question, matched_question):
    """Store a new question with the same answer as its matched question."""
    if matched_question in knowledge_base:
        answer = knowledge_base[matched_question]
        knowledge_base[original_question] = answer
        save_knowledge_base(knowledge_base)
        print(f"Stored similar question: {original_question} -> {answer}")

def get_response(knowledge_base, question):
    normalized_question = normalize_input(question)
    question_context.current_question = normalized_question

    if normalized_question == "no":
        return {
            "type": "learning",
            "response": "What is the correct answer?"
        }

    if not knowledge_base:
        question_context.original_question = normalized_question
        question_context.waiting_for_answer = True
        return {
            "type": "learning",
            "response": "I don't know the answer to that. Can you teach me?"
        }

    matches = process.extract(normalized_question, knowledge_base.keys(), limit=2)
    if not matches:
        question_context.original_question = normalized_question
        question_context.waiting_for_answer = True
        return {
            "type": "learning",
            "response": "I don't know the answer to that. Can you teach me?"
        }

    closest_question, closest_score = matches[0][0], matches[0][1]
    second_question, second_score = matches[1][0], matches[1][1]

    fuzzy_threshold = 90
    score_difference_threshold = 10

    if closest_score >= fuzzy_threshold:
        question_context.last_question = closest_question
        question_context.original_question = normalized_question
        # Store the similar question if it's not already in the knowledge base
        if normalized_question not in knowledge_base:
            store_similar_question(knowledge_base, normalized_question, closest_question)
        return {
            "type": "answer",
            "response": knowledge_base[closest_question]
        }

    if abs(closest_score - second_score) < score_difference_threshold:
        question_context.suggested_questions = [closest_question, second_question]
        question_context.is_clarifying = True
        question_context.original_question = normalized_question
        return {
            "type": "clarification",
            "response": f"Did you mean one of these?\n1. {closest_question}\n2. {second_question}\nPlease reply with '1', '2', or 'no'."
        }

    question_context.last_question = closest_question
    question_context.original_question = normalized_question
    return {
        "type": "answer",
        "response": knowledge_base[closest_question]
    }

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

        if normalized_input == "no":
            if question_context.original_question:
                question_context.waiting_for_answer = True
                question_context.needs_correction = True
                return jsonify({
                    "type": "learning",
                    "response": "What is the correct answer?"
                })
            else:
                return jsonify({
                    "type": "error",
                    "response": "I'm not sure what you're saying 'no' to. Please ask your question again."
                })

        if question_context.waiting_for_answer and question_context.needs_correction:
            if question_context.original_question:
                update_answer(knowledge_base, question_context.original_question, user_input)
                question_context.waiting_for_answer = False
                question_context.needs_correction = False
                question_context.original_question = None
                return jsonify({
                    "type": "success",
                    "response": "Thank you! I've updated my knowledge with the correct answer. 💖"
                })

        if question_context.is_clarifying:
            if normalized_input in ["1", "2"]:
                selected_index = int(normalized_input) - 1
                selected_question = question_context.suggested_questions[selected_index]
                
                # Store the original question with the same answer as the selected question
                if question_context.original_question and question_context.original_question != selected_question:
                    store_similar_question(knowledge_base, question_context.original_question, selected_question)
                
                question_context.last_question = selected_question
                question_context.is_clarifying = False
                return jsonify({
                    "type": "answer",
                    "response": knowledge_base[selected_question]
                })
            elif normalized_input == "no":
                question_context.waiting_for_answer = True
                question_context.needs_correction = True
                return jsonify({
                    "type": "learning",
                    "response": "What is the correct answer?"
                })

        if question_context.waiting_for_answer and not question_context.needs_correction:
            update_answer(knowledge_base, question_context.original_question, user_input)
            question_context.waiting_for_answer = False
            question_context.original_question = None
            question_context.possible_matches = []
            question_context.current_question = None
            question_context.suggested_questions = []
            question_context.last_question = None
            question_context.is_clarifying = False
            return jsonify({
                "type": "success",
                "response": "Thank you! I've learned something new. 💖"
            })
        response = get_response(knowledge_base, user_input)
        return jsonify(response)

    except Exception as e:
        print(f"Error in get_bot_response: {str(e)}")
        return jsonify({
            "type": "error",
            "response": "I encountered an error. Please try again."
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
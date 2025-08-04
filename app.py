import os
import sys
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from collections import defaultdict
from heapq import nlargest
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# --- Flask and MongoDB Setup ---
app = Flask(__name__)
CORS(app)

# MongoDB Atlas Connection
MONGO_URI = "mongodb+srv://<username>:<password>@cluster0.0hkywqp.mongodb.net/?retryWrites=true&w=majority"
DATABASE_NAME = "nlp_projects_db"
COLLECTION_NAME = "vip_summaries"
USERS_COLLECTION = "users"

# Gemini API Configuration
API_KEY = "<Gemini api key>"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Database Initialization ---
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    users_collection = db[USERS_COLLECTION]
    client.admin.command('ismaster')
    print("Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}", file=sys.stderr)
    sys.exit(1)

# --- Helper Functions ---
def validate_email(email):
    """Validate email format"""
    return '@' in email and '.' in email.split('@')[1]

def validate_password(password):
    """Validate password length"""
    return len(password) >= 6

def local_summarize(text, max_sentences=3):
    """Generate extractive summary using NLTK"""
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
        
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(text.lower())
    
    frequency = defaultdict(int)
    for word in words:
        if word not in stop_words:
            frequency[word] += 1
    
    max_frequency = max(frequency.values()) if frequency else 1
    for word in frequency:
        frequency[word] /= max_frequency
    
    sentence_scores = defaultdict(int)
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                sentence_scores[i] += frequency[word]
    
    top_sentences = nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join([sentences[i] for i in sorted(top_sentences)])
    
    return summary

def gemini_summarize(text):
    """Generate abstractive summary using Gemini API"""
    if not API_KEY:
        raise ValueError("Gemini API key is not set")

    prompt = f"Please provide a concise summary of the following text in 2-3 sentences:\n\n{text}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }]
    }

    for i in range(3):  # Retry up to 3 times
        try:
            response = requests.post(f"{API_URL}?key={API_KEY}", json=payload)
            response.raise_for_status()
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"API call failed (attempt {i+1}): {e}", file=sys.stderr)
            time.sleep(2 ** i)

    raise Exception("Failed to get summary from Gemini API")

# --- Routes ---
@app.route('/signup', methods=['POST'])
def signup():
    """Handle user registration"""
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required."}), 400

        email = data['email'].lower().strip()
        password = data['password']

        if not validate_email(email):
            return jsonify({"error": "Please enter a valid email address."}), 400

        if not validate_password(password):
            return jsonify({"error": "Password must be at least 6 characters long."}), 400

        if users_collection.find_one({"email": email}):
            return jsonify({"error": "Email already registered. Please login instead."}), 400

        user_id = users_collection.insert_one({
            "email": email,
            "password": generate_password_hash(password),
            "created_at": time.time()
        }).inserted_id

        return jsonify({
            "message": "User created successfully!",
            "userId": str(user_id),
            "email": email
        }), 201

    except Exception as e:
        print(f"Error in signup: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required."}), 400

        email = data['email'].lower().strip()
        password = data['password']

        user = users_collection.find_one({"email": email})
        if not user:
            return jsonify({"error": "Email not found. Please sign up first."}), 404

        if not check_password_hash(user['password'], password):
            return jsonify({"error": "Incorrect password."}), 401

        return jsonify({
            "message": "Login successful!",
            "userId": str(user['_id']),
            "email": email
        }), 200

    except Exception as e:
        print(f"Error in login: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Fetch user's summary history"""
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({"error": "User ID is required."}), 400

        if not users_collection.find_one({"_id": ObjectId(user_id)}):
            return jsonify({"error": "User not found."}), 404

        summaries = list(collection.find(
            {"user_id": user_id},
            {"_id": 0, "original_text": 1, "summary": 1, "timestamp": 1, "method": 1}
        ).sort("timestamp", -1).limit(20))

        return jsonify(summaries), 200

    except Exception as e:
        print(f"Error fetching history: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """Handle text summarization with hybrid approach"""
    try:
        data = request.get_json()
        if not data or 'text' not in data or 'userId' not in data:
            return jsonify({"error": "Text and user ID are required."}), 400

        user_id = data['userId']
        if not users_collection.find_one({"_id": ObjectId(user_id)}):
            return jsonify({"error": "User not found. Please login again."}), 404

        original_text = data['text'].strip()
        if len(original_text) < 20:
            return jsonify({"error": "Text must be at least 20 characters long."}), 400

        use_gemini = data.get('use_gemini', len(original_text) > 1000)
        
        try:
            if use_gemini and API_KEY:
                summary = gemini_summarize(original_text)
                method = "gemini"
            else:
                summary = local_summarize(original_text)
                method = "local"
        except Exception as e:
            summary = local_summarize(original_text)
            method = "local_fallback"
            print(f"Primary summarization failed, using fallback: {e}")

        document = {
            "user_id": user_id,
            "original_text": original_text,
            "summary": summary,
            "method": method,
            "timestamp": time.time()
        }
        collection.insert_one(document)

        return jsonify({
            "summary": summary,
            "method": method
        }), 200

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

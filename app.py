from flask import Flask, render_template, request, redirect, session, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import pdfplumber
import re
import whisper
import google.generativeai as genai
import json
from datetime import datetime

# =========================
# LOAD ENV VARIABLES
# =========================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Load Whisper model once
whisper_model = whisper.load_model("base")

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# =========================
# File Upload Config
# =========================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# =========================
# Database configuration
# =========================
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =========================
# Database Model
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), nullable=False)

with app.app_context():
    db.create_all()


# =========================
# Resume Analysis
# =========================
SKILLS_DB = {
    "python": ["Backend Developer", "Data Analyst"],
    "java": ["Software Developer"],
    "react": ["Frontend Developer"],
    "html": ["Frontend Developer"],
    "css": ["Frontend Developer"],
    "javascript": ["Full Stack Developer"],
    "sql": ["Database Administrator", "Data Analyst"],
    "machine learning": ["Data Scientist"],
    "flask": ["Backend Developer"],
    "django": ["Backend Developer"]
}

def analyze_resume(file_path):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    text = text.lower()

    detected_skills = []
    recommended_roles = set()

    for skill in SKILLS_DB:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            detected_skills.append(skill)
            for role in SKILLS_DB[skill]:
                recommended_roles.add(role)

    ats_score = min(len(detected_skills) * 15, 100)

    return detected_skills, list(recommended_roles), ats_score


# =========================
# ROUTES
# =========================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        if User.query.filter_by(email=email).first():
            flash("Email already exists.")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, password=hashed_password, role=role)

        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully.")
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            return redirect(url_for('dashboard'))

        flash("Invalid credentials.")
        return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('dashboard.html', user_name=session['user_name'])


# =========================
# Resume Upload
# =========================
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['resume']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            skills, roles, score = analyze_resume(filepath)

            session['skills'] = skills
            session['roles'] = roles
            session['ats_score'] = score

            return redirect(url_for('analysis_result'))

        flash("Only PDF files allowed.")

    return render_template('upload.html')


@app.route('/analysis_result')
def analysis_result():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template(
        'analysis_result.html',
        skills=session.get('skills', []),
        roles=session.get('roles', []),
        ats_score=session.get('ats_score', 0)
    )


# =========================
# Interview Flow
# =========================
@app.route('/start_interview', methods=['POST'])
def start_interview():
    selected_role = request.form.get('selected_role')
    session['selected_role'] = selected_role

    QUESTIONS_DB = {
        "Frontend Developer": [
            "Explain the virtual DOM.",
            "What is CSS Flexbox?"
        ],
        "Backend Developer": [
            "What is REST API?",
            "Explain Flask architecture."
        ]
    }

    session['questions'] = QUESTIONS_DB.get(selected_role, ["Tell me about yourself."])

    return redirect(url_for('interview'))


@app.route('/interview')
def interview():
    return render_template(
        'interview.html',
        role=session.get('selected_role'),
        questions=session.get('questions', [])
    )


# =========================
# AUDIO + AI EVALUATION
# =========================
@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return "No audio received", 400

    audio_file = request.files['audio']

    filename = f"recording_{datetime.now().strftime('%Y%m%d%H%M%S')}.webm"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    audio_file.save(filepath)

    # 1️⃣ Transcribe using Whisper
    result = whisper_model.transcribe(filepath, fp16=False)
    transcript = result["text"]
    print("TRANSCRIPT:", transcript)
    if not transcript.strip():
        session['technical_score'] = 0
        session['communication_score'] = 0
        session['confidence_score'] = 0
        session['ai_feedback'] = "No valid answer detected."
        return "Processed", 200

    # 2️⃣ Evaluate using Gemini
    prompt = f"""
    You are an AI Interview Evaluator.

    Evaluate the following candidate answer:

    "{transcript}"

    Respond ONLY in JSON:
    {{
      "technical_score": number (0-100),
      "communication_score": number (0-100),
      "confidence_score": number (0-100),
      "suggestions": "short advice"
    }}
    """

    response = model.generate_content(prompt)
    ai_text = response.text

    try:
        parsed = json.loads(ai_text)

        session['technical_score'] = parsed["technical_score"]
        session['communication_score'] = parsed["communication_score"]
        session['confidence_score'] = parsed["confidence_score"]
        session['ai_feedback'] = parsed["suggestions"]

    except:
        session['technical_score'] = 70
        session['communication_score'] = 70
        session['confidence_score'] = 70
        session['ai_feedback'] = "AI evaluation parsing failed."

    return "Processed", 200


@app.route('/feedback')
def feedback():
    return render_template(
        'feedback.html',
        technical=session.get('technical_score', 0),
        communication=session.get('communication_score', 0),
        confidence=session.get('confidence_score', 0),
        ai_feedback=session.get('ai_feedback', "")
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run()
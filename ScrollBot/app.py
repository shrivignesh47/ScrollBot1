# from flask import Flask, render_template, request, redirect, url_for, session
# from dotenv import load_dotenv
# import os

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate



# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# app = Flask(__name__)
# app.secret_key = "your_secret_key"  # Use a strong secret key

# # Dummy user store for example
# users = {}  # You can use a DB or file later


# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
#     the provided context, just say "answer is not available in the context". Don't guess.\n\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain


# def get_response(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]


# @app.route('/')
# def home():
#     return redirect(url_for('login'))


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users and users[uname] == pwd:
#             session['user'] = uname
#             return redirect(url_for('chatbot'))
#         else:
#             error = "Invalid username or password"
#     return render_template('login.html', error=error)


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users:
#             return "User already exists"
#         users[uname] = pwd
#         return redirect(url_for('login'))
#     return render_template('register.html')
# from markupsafe import Markup

# def format_answer(text):
#     formatted = ""
#     for line in text.split("\n"):
#         line = line.strip()
#         if line.startswith("*"):
#             formatted += f"<li>{line[1:].strip()}</li>"
#         elif line:
#             formatted += f"<p>{line}</p>"
#     if "<li>" in formatted:
#         formatted = "<ul>" + formatted + "</ul>"
#     return Markup(formatted)

# def get_pdf_answer(question):
#     raw_answer = get_response(question)
#     return format_answer(raw_answer)

# @app.route('/chatbot', methods=['GET', 'POST'])
# def chatbot():
#     if 'chat_history' not in session:
#         session['chat_history'] = []

#     if request.method == 'POST':
#         question = request.form['question']
#         answer = get_pdf_answer(question)


#         session['chat_history'].append({'question': question, 'answer': answer})
#         session.modified = True

#     return render_template('chatbot.html', chat_history=session.get('chat_history', []))
# @app.route('/clear_chat')
# def clear_chat():
#     session.pop('chat_history', None)  # Removes saved chat history from session
#     return redirect(url_for('chatbot'))  # Redirects back to chatbot page


# @app.route('/logout')
# def logout():
#     session.pop('user', None)
#     return redirect(url_for('login'))


# if __name__ == '__main__':
#     app.run(debug=True)


# app.py




# from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
# from dotenv import load_dotenv
# import os
# import re
# import json
# import logging
# from datetime import datetime
# from markupsafe import Markup, escape
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# import google.generativeai as genai

# # =============== 1. CONFIGURATION & INITIALIZATION ===============
# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = os.getenv("SECRET_KEY", "your_very_strong_secret_key_2025")

# # Paths
# FAISS_INDEX_PATH = "faiss_index"
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)
# CHAT_HISTORY_DIR = "user_chats"
# os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# # Set up logging
# logging.basicConfig(
#     filename=os.path.join(LOG_DIR, "medical_bot.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # Dummy user store (replace with database like SQLite in production)
# users = {}

# # Medical symptom synonym mapping for better understanding
# SYNONYM_MAP = {
#     "stomach pain": ["tummy ache", "belly pain", "gastric pain", "abdominal discomfort"],
#     "fever": ["high temperature", "running a fever", "hot body", "chills"],
#     "rash": ["skin rash", "red spots", "itchy skin", "hives"],
#     "headache": ["head pain", "migraine", "throbbing head", "pressure in head"],
#     "nausea": ["feeling sick", "want to vomit", "queasy", "upset stomach"],
#     "cough": ["dry cough", "wet cough", "persistent cough"],
#     "allergy": ["allergic reaction", "sneezing", "runny nose", "itchy eyes"]
# }

# def standardize_symptom(symptom: str) -> str:
#     """Map user-friendly terms to standardized medical terms."""
#     symptom = symptom.lower().strip()
#     for standard, variants in SYNONYM_MAP.items():
#         if symptom in variants or symptom == standard:
#             return standard
#     return symptom


# # =============== 2. MEDICAL INTENT DETECTION ===============
# def detect_medical_question(question: str) -> bool:
#     """Detect if a question is health-related using keyword matching."""
#     question = question.lower().strip()
#     medical_keywords = [
#         "pain", "fever", "sick", "allergy", "rash", "hurt", "symptom",
#         "medicine", "medication", "pill", "tablet", "dose", "prescription",
#         "headache", "cough", "vomiting", "diarrhea", "nausea", "dizzy",
#         "fatigue", "chest pain", "shortness of breath", "bleeding",
#         "doctor", "treatment", "infection", "swelling", "burning", "itch"
#     ]
#     return any(keyword in question for keyword in medical_keywords)


# # =============== 3. MEDICAL ADVICE GENERATION (AI + JSON) ===============
# def get_medical_advice_model():
#     """
#     Create a prompt that outputs strict JSON.
#     ✅ Uses {{ }} to escape literal braces in JSON schema (fixes INVALID_PROMPT_INPUT)
#     """
#     prompt_template = """
# You are a virtual health assistant. Generate a structured medical guidance response in **strict JSON format only**.
# Do not add any text before or after.

# Use this schema:
# {{  "greeting": "Personalized thank-you message with name and age",
#     "summary": "Brief summary of possible causes (non-diagnostic)",
#     "recommendations": ["List", "of", "actionable", "steps"],
#     "action": "When to consult a doctor"
# }}

# Guidelines:
# - Be empathetic and professional.
# - Never claim to diagnose.
# - Keep recommendations safe and general.
# - Use only the information provided.

# Patient Info:
# Name: {name}
# Age: {age}
# Symptoms: {symptoms}

# Respond ONLY with valid JSON.
# """
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, convert_system_message_to_human=True)
#     return prompt | model


# def generate_medical_advice(name, age, symptoms):
#     """Call AI, parse JSON, and return formatted HTML advice."""
#     try:
#         chain = get_medical_advice_model()
#         standardized_symptoms = ", ".join(standardize_symptom(s.strip()) for s in symptoms.split(",") if s.strip())
        
#         response = chain.invoke({
#             "name": escape(name),
#             "age": age,
#             "symptoms": standardized_symptoms
#         })

#         text = response.content.strip()

#         # Remove markdown wrappers
#         if text.startswith("```json"):
#             text = text[7:-3] if text.endswith("```") else text[7:]
#         elif text.startswith("```"):
#             text = text[3:-3]

#         data = json.loads(text)

#         # Build rich, safe HTML response
#         html = f"<div class='medical-advice-card'>"
#         html += f"<b>🩺 {escape(data['greeting'])}</b><br><br>"
#         if data.get("summary"):
#             html += f"<i>{escape(data['summary'])}</i><br><br>"
#         if data.get("recommendations"):
#             html += "<b>💡 Recommended Steps:</b><ul class='steps'>"
#             for item in data["recommendations"]:
#                 html += f"<li>{escape(item)}</li>"
#             html += "</ul><br>"
#         if data.get("action"):
#             html += f"<b>⚠️ Important:</b> <em>{escape(data['action'])}</em>"
#         html += "</div>"

#         # Log anonymized interaction
#         logging.info(f"Medical Advice | User: {session.get('user')} | Age: {age} | Symptoms: {standardized_symptoms}")

#         return Markup(html)

#     except json.JSONDecodeError as e:
#         logging.warning(f"JSON Parse Failed (1st try): {str(e)} | Retrying...")
#         try:
#             # Retry once
#             response = chain.invoke({
#                 "name": escape(name),
#                 "age": age,
#                 "symptoms": standardized_symptoms
#             })
#             text = response.content.strip().strip("```json").strip("`")
#             data = json.loads(text)
#             return build_html_from_data(data)
#         except Exception as retry_e:
#             logging.error(f"JSON retry failed: {str(retry_e)}")
#             return Markup("⚠️ I couldn't process your symptoms. Please consult a doctor.")
#     except Exception as e:
#         logging.error(f"Medical AI Error: {str(e)}")
#         return Markup("⚠️ I'm currently unable to assist. Please seek medical help if urgent.")


# def build_html_from_data(data):
#     """Helper to build HTML from parsed JSON (used in retry)."""
#     html = f"<b>🩺 {escape(data['greeting'])}</b><br><br>"
#     if data.get("summary"):
#         html += f"<i>{escape(data['summary'])}</i><br><br>"
#     if data.get("recommendations"):
#         html += "<b>💡 What You Can Do:</b><ul>"
#         for item in data["recommendations"]:
#             html += f"<li>{escape(item)}</li>"
#         html += "</ul><br>"
#     if data.get("action"):
#         html += f"<b>⚠️ Important:</b> {escape(data['action'])}"
#     return Markup(html)


# # =============== 4. PDF-BASED MEDICAL RESPONSE ===============
# def get_pdf_response(user_question):
#     """Retrieve from FAISS and answer using medical PDFs."""
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         try:
#             vector_store = FAISS.load_local(
#                 FAISS_INDEX_PATH,
#                 embeddings,
#                 allow_dangerous_deserialization=True  # ✅ Required to load
#             )
#         except Exception as e:
#             logging.error(f"Failed to load FAISS index: {str(e)}")
#             return "I'm unable to access the medical knowledge base."

#         retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#         docs = retriever.get_relevant_documents(user_question)

#         logging.info(f"Retrieved {len(docs)} docs for: {user_question}")

#         if not docs:
#             return "I couldn't find relevant information in the documents."

#         # Medical-only prompt to prevent hallucination
#         prompt_template = """
# Answer the medical question using **only** the context below.
# If not relevant, say: 'I can't find specific details about this in the current knowledge base.'

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#         prompt = ChatPromptTemplate.from_template(prompt_template)
#         model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#         chain = create_stuff_documents_chain(llm=model, prompt=prompt)

#         response = chain.invoke({
#             "context": docs,
#             "question": user_question
#         })

#         # Add citations
#         references = [
#             f'<sup>[{i+1}]</sup> p.{doc.metadata["page"]} ({os.path.basename(doc.metadata["source"])})'
#             for i, doc in enumerate(docs)
#         ]
#         final = response.strip()
#         if references:
#             final += "<br><br><strong>Sources:</strong> " + " ".join(references)

#         return final

#     except Exception as e:
#         logging.error(f"PDF Retrieval Error: {str(e)}")
#         return "I'm having trouble accessing the knowledge base right now."


# # =============== 5. SESSION & MEDICAL FLOW MANAGEMENT ===============
# def get_medical_data():
#     """Safely retrieve or initialize medical session data."""
#     if "medical_data" not in session:
#         session["medical_data"] = {"name": None, "age": None, "symptoms": []}
#         session.modified = True

#     data = session.get("medical_data", {})
#     if not isinstance(data, dict):
#         data = {"name": None, "age": None, "symptoms": []}

#     data.setdefault("name", None)
#     data.setdefault("age", None)
#     if not isinstance(data.get("symptoms"), list):
#         data["symptoms"] = []

#     session["medical_data"] = data
#     session.modified = True
#     return data


# def handle_medical_question(question):
#     """Handle multi-turn medical conversation, but allow switching topics."""
#     data = get_medical_data()
#     lower_q = question.strip().lower()

#     # Check if this is a NEW medical topic (e.g., user says "cardiovascular" mid-flow)
#     if detect_medical_question(lower_q):
#         # List of keywords that suggest a shift to general medical query
#         general_medical_topics = [
#             "dvt", "cardiovascular", "ulcerative colitis", "diabetes",
#             "hypertension", "asthma", "arthritis", "cancer", "infection",
#             "treatment for", "what is", "symptoms of", "causes of"
#         ]
#         if any(topic in lower_q for topic in general_medical_topics):
#             # ✅ User is switching topic → exit current flow
#             session.pop("medical_data", None)
#             session.pop("last_medical_trigger", None)
#             session.modified = True
#             # Now answer from PDF
#             return get_pdf_response(question)

#     # If full data is collected
#     if data["name"] and data["age"] and data["symptoms"]:
#         symptoms_str = ", ".join(data["symptoms"])
#         if lower_q in ["repeat", "show", "details"]:
#             return (
#                 f"<b>📋 Your Details:</b><br>"
#                 f"• <b>Name:</b> {data['name']}<br>"
#                 f"• <b>Age:</b> {data['age']}<br>"
#                 f"• <b>Symptoms:</b> {symptoms_str}<br><br>"
#                 "Say <b>update</b> to change, or <b>no</b> if you're done."
#             )
#         if lower_q in ["update", "change"]:
#             session["medical_data"] = {"name": None, "age": None, "symptoms": []}
#             session.modified = True
#             return "Let's start over. What's your name?"
#         if lower_q == "no":
#             session.pop("medical_data", None)
#             return "Got it. Feel free to ask anything else!"
#         if detect_medical_question(lower_q):
#             std_symptom = standardize_symptom(question)
#             existing_std = [standardize_symptom(s) for s in data["symptoms"]]
#             if std_symptom not in existing_std:
#                 data["symptoms"].append(question.strip())
#                 session["medical_data"] = data
#                 session.modified = True
#             return generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))
#         return "⚠️ I can only assist with medical follow-ups."

#     # Step 1: Get name
#     if data["name"] is None:
#         name = question.strip()
#         if len(name) < 2 or not re.match(r"^[A-Za-z\s]+$", name):
#             return "Please enter a valid name (letters only)."
#         data["name"] = name
#         session["medical_data"] = data
#         session.modified = True
#         return "Thank you! What's your age?"

#     # Step 2: Get age
#     if data["age"] is None:
#         if re.match(r"^\d+$", question.strip()):
#             age_val = int(question.strip())
#             if 1 <= age_val <= 120:
#                 data["age"] = age_val
#                 session["medical_data"] = data
#                 session.modified = True
#                 return "Got it. Please describe your symptoms clearly (e.g., fever, nausea)."
#             else:
#                 return "Please enter a realistic age (1–120)."
#         return "Enter a number."

#     # Step 3: Collect symptoms
#     std_symptom = standardize_symptom(question)
#     existing_std = [standardize_symptom(s) for s in data["symptoms"]]
#     if std_symptom not in existing_std:
#         data["symptoms"].append(question.strip())
#         session["medical_data"] = data
#         session.modified = True

#     return generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))

# # =============== 6. ROUTES ===============
# @app.route('/')
# def home():
#     return redirect(url_for('login'))


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users and users[uname] == pwd:
#             session['user'] = uname
#             session['chat_history'] = []
#             flash("Logged in successfully!", "success")
#             return redirect(url_for('chatbot'))
#         else:
#             error = "Invalid username or password"
#     return render_template('login.html', error=error)


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users:
#             return "User already exists. <a href='/login'>Login</a>."
#         users[uname] = pwd
#         flash("Registration successful!", "info")
#         return redirect(url_for('login'))
#     return render_template('register.html')

# @app.route('/chatbot', methods=['GET', 'POST'])
# def chatbot():
#     if 'user' not in session:
#         return redirect(url_for('login'))

#     if 'chat_history' not in session:
#         session['chat_history'] = []

#     if request.method == 'POST':
#         question = request.form.get('question', '').strip()
#         lower_q = question.lower().strip()

#         if not question:
#             answer = "Please enter a message."
#         else:
#             if lower_q in ["hi", "hello", "hey"]:
#                 answer = "Hi 👋 I'm your Medical Assistant. Please describe your symptoms."

#             # ✅ Always allow switching to general medical topic
#             elif detect_medical_question(lower_q) and any(
#                 kw in lower_q for kw in ["what is", "treatment for", "symptoms of", "causes of", 
#                                         "dvt", "cardiovascular", "ulcerative colitis", "diabetes"]
#             ):
#                 # Exit medical flow and answer from PDF
#                 session.pop('medical_data', None, None)
#                 session.pop('last_medical_trigger', None, None)
#                 answer = get_pdf_response(question)

#             elif detect_medical_question(lower_q) or "medical_data" in session:
#                 session["last_medical_trigger"] = question.strip()
#                 answer = handle_medical_question(question)

#             else:
#                 answer = get_pdf_response(question)  # Fallback to PDF

#         # Save to chat history
#         session['chat_history'].append({
#             'question': escape(question),
#             'answer': answer
#         })
#         session.modified = True

#     return render_template('chatbot.html', chat_history=session['chat_history'])

# @app.route('/clear_chat')
# def clear_chat():
#     session.pop('chat_history', None)
#     session.pop('medical_data', None)
#     session.pop('last_medical_trigger', None)
#     flash("Chat cleared! 🧹", "info")
#     return redirect(url_for('chatbot'))


# @app.route('/logout')
# def logout():
#     username = session.get('user', 'User')
#     session.clear()
#     flash(f"See you later, {username}! 👋", "info")
#     return redirect(url_for('login'))


# # =============== 7. LAUNCH APP ===============
# if __name__ == '__main__':
#     print("✅ Medical Chatbot is running on http://127.0.0.1:5000")
#     app.run(debug=True) 


# from flask import Flask, render_template, request, redirect, url_for, session, flash
# from dotenv import load_dotenv
# import os
# import re
# import json
# import logging
# from markupsafe import Markup, escape
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# import google.generativeai as genai

# # =============== 1. CONFIGURATION ===============
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# app = Flask(__name__)
# app.secret_key = os.getenv("SECRET_KEY", "your_very_strong_secret_key_2025")

# # Paths
# FAISS_INDEX_PATH = "faiss_index"
# LOG_DIR = "logs"
# os.makedirs(LOG_DIR, exist_ok=True)

# # Logging
# logging.basicConfig(
#     filename=os.path.join(LOG_DIR, "medical_bot.log"),
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # Dummy users
# users = {}

# # Symptom synonyms
# SYNONYM_MAP = {
#     "stomach pain": ["tummy ache", "belly pain", "gastric pain", "abdominal discomfort"],
#     "fever": ["high temperature", "running a fever", "hot body", "chills"],
#     "rash": ["skin rash", "red spots", "itchy skin"],
#     "headache": ["head pain", "migraine", "throbbing head"],
#     "nausea": ["feeling sick", "want to vomit", "queasy"]
# }

# def standardize_symptom(symptom: str) -> str:
#     s = symptom.lower().strip()
#     for std, variants in SYNONYM_MAP.items():
#         if s in variants or s == std:
#             return std
#     return s


# # =============== 2. MEDICAL INTENT DETECTION ===============
# def detect_medical_question(q: str) -> bool:
#     keywords = [
#         "pain", "fever", "sick", "allergy", "rash", "hurt", "symptom",
#         "medicine", "headache", "cough", "vomiting", "nausea", "dizzy",
#         "fatigue", "chest pain", "shortness of breath", "bleeding",
#         "doctor", "treatment", "infection", "swelling", "burning", "itch"
#     ]
#     q = q.lower()
#     return any(k in q for k in keywords)


# # =============== 3. DETECT GENERAL MEDICAL KNOWLEDGE QUERY ===============
# def is_general_medical_query(question: str) -> bool:
#     """Detect if user is asking a broad medical knowledge question."""
#     general_phrases = [
#         "what is", "treatment for", "symptoms of", "causes of", "define",
#         "explain", "how to treat", "management of", "diagnosis of"
#     ]
#     specific_conditions = [
#         "dvt", "deep venous thrombosis", "cardiovascular", "ulcerative colitis",
#         "diabetes", "hypertension", "asthma", "arthritis", "cancer", "infection",
#         "pneumonia", "migraine", "stroke", "heart attack", "angina"
#     ]
#     q = question.lower().strip()
#     return (
#         any(phrase in q for phrase in general_phrases) or
#         any(cond in q for cond in specific_conditions)
#     )


# # =============== 4. PDF-BASED RESPONSE ===============
# def get_pdf_response(user_question):
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#         vector_store = FAISS.load_local(
#             FAISS_INDEX_PATH,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )
#         retriever = vector_store.as_retriever(search_kwargs={"k": 3})
#         docs = retriever.get_relevant_documents(user_question)

#         if not docs:
#             return "I couldn't find relevant information in the documents."

#         prompt_template = """
# Answer the medical question using **only** the context below.
# If not relevant, say: 'I can't find specific details about this in the current knowledge base.'

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#         prompt = ChatPromptTemplate.from_template(prompt_template)
#         model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
#         chain = create_stuff_documents_chain(llm=model, prompt=prompt)
#         response = chain.invoke({"context": docs, "question": user_question})

#         references = [
#             f'<sup>[{i+1}]</sup> p.{doc.metadata["page"]} ({os.path.basename(doc.metadata["source"])})'
#             for i, doc in enumerate(docs)
#         ]
#         final = response.strip()
#         if references:
#             final += "<br><br><strong>Sources:</strong> " + " ".join(references)

#         return final

#     except Exception as e:
#         logging.error(f"PDF Error: {str(e)}")
#         return "I'm having trouble accessing the knowledge base right now."


# # =============== 5. AI MEDICAL ADVICE (JSON) ===============
# def get_medical_advice_model():
#     # ✅ Fixed: {{ }} escapes JSON schema to avoid INVALID_PROMPT_INPUT
#     prompt_template = """
# You are a virtual health assistant. Generate a structured medical guidance response in **strict JSON format only**.
# Do not add any text before or after.

# Use this schema:
# {{  "greeting": "Personalized thank-you message with name and age",
#     "summary": "Brief summary of possible causes (non-diagnostic)",
#     "recommendations": ["List", "of", "actionable", "steps"],
#     "action": "When to consult a doctor"
# }}

# Guidelines:
# - Be empathetic and professional.
# - Never claim to diagnose.
# - Keep recommendations safe and general.

# Patient Info:
# Name: {name}
# Age: {age}
# Symptoms: {symptoms}

# Respond ONLY with valid JSON.
# """
#     prompt = ChatPromptTemplate.from_template(prompt_template)
#     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, convert_system_message_to_human=True)
#     return prompt | model


# def generate_medical_advice(name, age, symptoms):
#     try:
#         chain = get_medical_advice_model()
#         standardized_symptoms = ", ".join(standardize_symptom(s.strip()) for s in symptoms.split(",") if s.strip())
        
#         response = chain.invoke({
#             "name": escape(name),
#             "age": age,
#             "symptoms": standardized_symptoms
#         })
#         text = response.content.strip()

#         if text.startswith("```json"):
#             text = text[7:-3] if text.endswith("```") else text[7:]
#         elif text.startswith("```"):
#             text = text[3:-3]

#         data = json.loads(text)

#         html = f"<b>🩺 {escape(data['greeting'])}</b><br><br>"
#         if data.get("summary"):
#             html += f"<i>{escape(data['summary'])}</i><br><br>"
#         if data.get("recommendations"):
#             html += "<b>💡 What You Can Do:</b><ul>"
#             for item in data["recommendations"]:
#                 html += f"<li>{escape(item)}</li>"
#             html += "</ul><br>"
#         if data.get("action"):
#             html += f"<b>⚠️ Important:</b> {escape(data['action'])}"

#         return Markup(html)

#     except Exception as e:
#         logging.error(f"Medical AI Error: {str(e)}")
#         return Markup("⚠️ I'm currently unable to assist. Please consult a healthcare provider.")


# # =============== 6. SESSION MANAGEMENT ===============
# def get_medical_data():
#     if "medical_data" not in session:
#         session["medical_data"] = {"name": None, "age": None, "symptoms": []}
#         session.modified = True

#     data = session["medical_data"]
#     if not isinstance(data, dict):
#         data = {"name": None, "age": None, "symptoms": []}

#     data.setdefault("name", None)
#     data.setdefault("age", None)
#     if not isinstance(data.get("symptoms"), list):
#         data["symptoms"] = []

#     session["medical_data"] = data
#     session.modified = True
#     return data

# def handle_medical_question(question):
#     data = get_medical_data()
#     lower_q = question.strip().lower()

#     # 🔥 Always check for general knowledge queries first
#     if is_general_medical_query(lower_q):
#         session.pop("medical_data", None)
#         session.pop("last_medical_trigger", None)
#         session.modified = True
#         return get_pdf_response(question)

#     # ✅ If full data already collected
#     if data["name"] and data["age"] and data["symptoms"]:
#         symptoms_str = ", ".join(data["symptoms"])

#         if lower_q in ["repeat", "show", "details"]:
#             return (
#                 f"<b>📋 Your Details:</b><br>"
#                 f"• <b>Name:</b> {data['name']}<br>"
#                 f"• <b>Age:</b> {data['age']}<br>"
#                 f"• <b>Symptoms:</b> {symptoms_str}<br><br>"
#                 "Say <b>update</b> to change, or <b>no</b> if done."
#             )

#         if lower_q in ["update", "change"]:
#             session["medical_data"] = {"name": None, "age": None, "symptoms": []}
#             session.modified = True
#             return "Let's start over. What's your name?"

#         if lower_q == "no":
#             session.pop("medical_data", None)
#             return "Got it. Feel free to ask anything else!"

#         if detect_medical_question(question):
#             std_symptom = standardize_symptom(question)
#             existing_std = [standardize_symptom(s) for s in data["symptoms"]]
#             if std_symptom not in existing_std:
#                 data["symptoms"].append(question.strip())
#                 session["medical_data"] = data
#                 session.modified = True
#             return generate_medical_advice(
#                 data["name"], data["age"], ", ".join(data["symptoms"])
#             )

#         return "⚠️ I can only assist with medical follow-ups or general medical knowledge."

#     # ✅ Step 1: Get name (extra validation to avoid 'fever' etc. as names)
#     if data["name"] is None:
#         name = question.strip()
#         # block common symptom/condition words from being names
#         blocked_terms = ["fever", "cough", "pain", "rash", "nausea", "headache"]
#         if len(name) < 2 or not re.match(r"^[A-Za-z\s]+$", name) or name.lower() in blocked_terms:
#             return "Please enter your actual name (letters only, not a symptom)."
#         data["name"] = name
#         session["medical_data"] = data
#         session.modified = True
#         return "Thank you! What's your age?"

#     # ✅ Step 2: Get age
#     if data["age"] is None:
#         if re.match(r"^\d+$", question.strip()):
#             age_val = int(question.strip())
#             if 1 <= age_val <= 120:
#                 data["age"] = age_val
#                 session["medical_data"] = data
#                 session.modified = True
#                 return "Got it. Please describe your symptoms clearly (e.g., fever, nausea)."
#             else:
#                 return "Please enter a realistic age (1–120)."
#         return "Enter a number."

#     # ✅ Step 3: Collect symptoms
#     std_symptom = standardize_symptom(question)
#     existing_std = [standardize_symptom(s) for s in data["symptoms"]]
#     if std_symptom not in existing_std:
#         data["symptoms"].append(question.strip())
#         session["medical_data"] = data
#         session.modified = True

#     return generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))

# # =============== 7. ROUTES ===============
# @app.route('/')
# def home():
#     return redirect(url_for('login'))


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     error = None
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users and users[uname] == pwd:
#             session['user'] = uname
#             session['chat_history'] = []
#             flash("Logged in successfully!", "success")
#             return redirect(url_for('chatbot'))
#         else:
#             error = "Invalid username or password"
#     return render_template('login.html', error=error)


# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         uname = request.form['username']
#         pwd = request.form['password']
#         if uname in users:
#             return "User already exists. <a href='/login'>Login</a>."
#         users[uname] = pwd
#         flash("Registration successful!", "info")
#         return redirect(url_for('login'))
#     return render_template('register.html')

# @app.route('/chatbot', methods=['GET', 'POST'])
# def chatbot():
#     if 'user' not in session:
#         return redirect(url_for('login'))

#     if 'chat_history' not in session:
#         session['chat_history'] = []

#     if request.method == 'POST':
#         question = request.form.get('question', '').strip()
#         if not question:
#             answer = "Please enter a message."
#         else:
#             lower_q = question.lower().strip()

#             if lower_q in ["hi", "hello", "hey"]:
#                 answer = "Hi 👋 I'm your Medical Assistant. You can ask about symptoms or medical conditions."

#             # ✅ Check general query BEFORE starting medical flow
#             elif is_general_medical_query(lower_q):
#                 session.pop('medical_data', None)
#                 session.pop('last_medical_trigger', None)
#                 session.modified = True
#                 answer = get_pdf_response(question)

#             elif detect_medical_question(lower_q) or "medical_data" in session:
#                 session["last_medical_trigger"] = question.strip()
#                 answer = handle_medical_question(question)  # This now has escape logic

#             else:
#                 answer = get_pdf_response(question)

#         session['chat_history'].append({
#             'question': escape(question),
#             'answer': answer
#         })
#         session.modified = True

#     return render_template('chatbot.html', chat_history=session['chat_history'])

# @app.route('/clear_chat')
# def clear_chat():
#     session.pop('chat_history', None)
#     session.pop('medical_data', None)
#     session.pop('last_medical_trigger', None)
#     flash("Chat cleared! 🧹", "info")
#     return redirect(url_for('chatbot'))


# @app.route('/logout')
# def logout():
#     username = session.get('user', 'User')
#     session.clear()
#     flash(f"See you later, {username}! 👋", "info")
#     return redirect(url_for('login'))


# if __name__ == '__main__':
#     print("✅ Medical Chatbot running on http://127.0.0.1:5000")
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv
import os
import re
import json
import logging
from markupsafe import Markup, escape
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

# =============== 1. CONFIGURATION ===============
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your_very_strong_secret_key_2025")

# Paths
FAISS_INDEX_PATH = "faiss_index"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "medical_bot.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Dummy users
users = {}

# Symptom synonyms
SYNONYM_MAP = {
    "stomach pain": ["tummy ache", "belly pain", "gastric pain", "abdominal discomfort"],
    "fever": ["high temperature", "running a fever", "hot body", "chills"],
    "rash": ["skin rash", "red spots", "itchy skin"],
    "headache": ["head pain", "migraine", "throbbing head"],
    "nausea": ["feeling sick", "want to vomit", "queasy"]
}

def standardize_symptom(symptom: str) -> str:
    s = symptom.lower().strip()
    for std, variants in SYNONYM_MAP.items():
        if s in variants or s == std:
            return std
    return s


# =============== 2. MEDICAL INTENT DETECTION ===============
def detect_medical_question(q: str) -> bool:
    keywords = [
        "pain", "fever", "sick", "allergy", "rash", "hurt", "symptom",
        "medicine", "headache", "cough", "vomiting", "nausea", "dizzy",
        "fatigue", "chest pain", "shortness of breath", "bleeding",
        "doctor", "treatment", "infection", "swelling", "burning", "itch"
    ]
    q = q.lower()
    return any(k in q for k in keywords)


# =============== 3. DETECT GENERAL MEDICAL KNOWLEDGE QUERY ===============
def is_general_medical_query(question: str) -> bool:
    """Detect if user is asking a broad medical knowledge question."""
    general_phrases = [
        "what is", "treatment for", "symptoms of", "causes of", "define",
        "explain", "how to treat", "management of", "diagnosis of"
    ]
    specific_conditions = [
        "dvt", "deep venous thrombosis", "cardiovascular", "ulcerative colitis",
        "diabetes", "hypertension", "asthma", "arthritis", "cancer", "infection",
        "pneumonia", "migraine", "stroke", "heart attack", "angina"
    ]
    q = question.lower().strip()
    return (
        any(phrase in q for phrase in general_phrases) or
        any(cond in q for cond in specific_conditions)
    )


# =============== 4. GLOBAL RESOURCES (load once) ===============
# Embeddings and FAISS index loaded once to avoid repeated I/O
_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
try:
    _vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        _embeddings,
        allow_dangerous_deserialization=True
    )
    _retriever = _vector_store.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    logging.error(f"Failed to load FAISS index at startup: {e}")
    _vector_store = None
    _retriever = None

# Create reusable LLM instances with convert_system_message_to_human=False to silence the warning
PDF_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    convert_system_message_to_human=False
)

EXTERNAL_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.25,
    convert_system_message_to_human=False
)

ADVICE_LLM_PROMPT_WRAPPER = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    convert_system_message_to_human=False
)


# =============== 5. PDF-BASED RESPONSE (returns (text, found_bool)) ===============
def get_pdf_response(user_question):
    """
    Returns a tuple: (text_html, found_bool)
    found_bool True  -> meaningful answer found in PDF (text_html contains the answer)
    found_bool False -> not found (text_html may contain a not-found message)
    """
    try:
        if _retriever is None:
            return ("PDF index not available.", False)

        # Use the new invoke API on the retriever
        docs = _retriever.invoke(user_question)

        if not docs:
            return ("I couldn't find relevant information in the documents.", False)

        prompt_template = """
Answer the medical question using **only** the context below.
If not relevant, say: 'I can't find specific details about this in the current knowledge base.'

Context:
{context}

Question:
{question}

Answer:
"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = create_stuff_documents_chain(llm=PDF_LLM, prompt=prompt)
        response = chain.invoke({"context": docs, "question": user_question})

        final = response.strip()
        # If the model indicates "I can't find specific details..." treat as not found
        lowered = final.lower()
        if "i can't find specific details" in lowered or "i couldn't find" in lowered or len(final) < 10:
            return ("I couldn't find specific details about this in the current knowledge base.", False)

        # Build sources list for display
        references = [
            f'<sup>[{i+1}]</sup> p.{doc.metadata.get("page","?")} ({os.path.basename(doc.metadata.get("source","unknown"))})'
            for i, doc in enumerate(docs)
        ]
        if references:
            final += "<br><br><strong>Sources (from PDF):</strong> " + " ".join(references)

        return (final, True)

    except Exception as e:
        logging.error(f"PDF Error: {str(e)}")
        return ("I'm having trouble accessing the knowledge base right now.", False)


# =============== 6. EXTERNAL / GenAI-FALLBACK RESPONSE ===============
def get_external_response(user_question):
    """
    Query the generative model (or web) as a fallback.
    Return an HTML-safe string containing the response and sources.
    """
    try:
        prompt_template = f"""
You are a medical information assistant. Answer the user question concisely and include a short list of reliable sources (site names or organizations) at the end.
Question: {user_question}

Answer:
"""
        # Use EXTERNAL_LLM and invoke
        resp = EXTERNAL_LLM.invoke({"input": prompt_template})
        final = getattr(resp, "content", None) or str(resp)
        final = final.strip()

        # Ensure we append a source tag so the UI knows this came from web/genai
        if "source" not in final.lower() and "sources" not in final.lower():
            final += "<br><br><strong>Sources:</strong> Web / GenAI (not in PDF)"

        return final

    except Exception as e:
        logging.error(f"External GenAI Error: {str(e)}")
        return "I'm having trouble fetching external information right now."


# =============== 7. AI MEDICAL ADVICE (JSON) ===============
def get_medical_advice_model():
    # JSON-only schema model
    prompt_template = """
You are a virtual health assistant. Generate a structured medical guidance response in **strict JSON format only**.
Do not add any text before or after.

Use this schema:
{{  "greeting": "Personalized thank-you message with name and age",
    "summary": "Brief summary of possible causes (non-diagnostic)",
    "recommendations": ["List", "of", "actionable", "steps"],
    "action": "When to consult a doctor"
}}

Guidelines:
- Be empathetic and professional.
- Never claim to diagnose.
- Keep recommendations safe and general.

Patient Info:
Name: {name}
Age: {age}
Symptoms: {symptoms}

Respond ONLY with valid JSON.
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # Return a combined prompt+llm callable - we reuse ADVICE_LLM_PROMPT_WRAPPER implicitly via chain invocation
    return prompt | ADVICE_LLM_PROMPT_WRAPPER


def generate_medical_advice(name, age, symptoms):
    try:
        chain = get_medical_advice_model()
        standardized_symptoms = ", ".join(standardize_symptom(s.strip()) for s in symptoms.split(",") if s.strip())

        response = chain.invoke({
            "name": escape(name),
            "age": age,
            "symptoms": standardized_symptoms
        })
        text = getattr(response, "content", "") or str(response)
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:-3] if text.endswith("```") else text[7:]
        elif text.startswith("```"):
            text = text[3:-3]

        data = json.loads(text)

        html = f"<b>🩺 {escape(data['greeting'])}</b><br><br>"
        if data.get("summary"):
            html += f"<i>{escape(data['summary'])}</i><br><br>"
        if data.get("recommendations"):
            html += "<b>💡 What You Can Do:</b><ul>"
            for item in data["recommendations"]:
                html += f"<li>{escape(item)}</li>"
            html += "</ul><br>"
        if data.get("action"):
            html += f"<b>⚠️ Important:</b> {escape(data['action'])}"

        return Markup(html)

    except Exception as e:
        logging.error(f"Medical AI Error: {str(e)}")
        return Markup("⚠️ I'm currently unable to assist. Please consult a healthcare provider.")


# =============== 8. SESSION MANAGEMENT ===============
def get_medical_data():
    if "medical_data" not in session:
        session["medical_data"] = {"name": None, "age": None, "symptoms": []}
        session.modified = True

    data = session["medical_data"]
    if not isinstance(data, dict):
        data = {"name": None, "age": None, "symptoms": []}

    data.setdefault("name", None)
    data.setdefault("age", None)
    if not isinstance(data.get("symptoms"), list):
        data["symptoms"] = []

    session["medical_data"] = data
    session.modified = True
    return data


def handle_medical_question(question):
    """
    New flow:
    - Always check general medical query / PDF first.
    - If PDF found -> return PDF answer (with PDF sources).
    - If PDF not found -> fallback to external GenAI -> return external answer with note.
    - For personal symptom flow: collect name/age/symptoms, then produce personalized advice + knowledge lookup.
    """
    data = get_medical_data()
    lower_q = question.strip().lower()

    # 1) If it's a general medical knowledge question (explicit), then PDF-first -> external fallback
    if is_general_medical_query(lower_q):
        pdf_text, pdf_found = get_pdf_response(question)
        if pdf_found:
            # Return PDF answer with a clear source label
            return Markup(pdf_text + "<br><br><strong>📖 Source:</strong> Local PDF knowledge base")
        else:
            # Fallback to external GenAI/web
            ext = get_external_response(question)
            return Markup("I couldn't find specific details in the PDF.<br><br>" + ext)

    # 2) If user is already in the personal flow and all data collected -> produce personalized advice + KB lookup
    if data["name"] and data["age"] and data["symptoms"]:
        symptoms_str = ", ".join(data["symptoms"])

        if lower_q in ["repeat", "show", "details"]:
            return (
                f"<b>📋 Your Details:</b><br>"
                f"• <b>Name:</b> {escape(data['name'])}<br>"
                f"• <b>Age:</b> {data['age']}<br>"
                f"• <b>Symptoms:</b> {escape(symptoms_str)}<br><br>"
                "Say <b>update</b> to change, or <b>no</b> if done."
            )

        if lower_q in ["update", "change"]:
            session["medical_data"] = {"name": None, "age": None, "symptoms": []}
            session.modified = True
            return "Let's start over. What's your name?"

        if lower_q == "no":
            session.pop("medical_data", None)
            return "Got it. Feel free to ask anything else!"

        # If they asked another symptom-like input while in personal flow, add symptom and produce advice+KB lookup
        if detect_medical_question(question):
            std_symptom = standardize_symptom(question)
            existing_std = [standardize_symptom(s) for s in data["symptoms"]]
            if std_symptom not in existing_std:
                data["symptoms"].append(question.strip())
                session["medical_data"] = data
                session.modified = True

            # Produce personal advice
            personal_html = generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))

            # Now knowledge lookup (PDF-first -> external fallback) using combined symptoms as query
            query = ", ".join(data["symptoms"])
            pdf_text, pdf_found = get_pdf_response(query)
            if pdf_found:
                combined = str(personal_html) + "<hr>" + pdf_text + "<br><br><strong>📖 Source:</strong> Local PDF knowledge base"
                return Markup(combined)
            else:
                ext = get_external_response(query)
                combined = str(personal_html) + "<hr>" + "I couldn't find specific details in the PDF.<br><br>" + ext
                return Markup(combined)

        return "⚠️ I can only assist with medical follow-ups or general medical knowledge."

    # 3) Else - personal flow collecting steps: Name -> Age -> Symptoms
    # Note: we already checked is_general_medical_query at the top so "cardio vascular" won't be swallowed here.

    # Step 1: Get name (extra validation to avoid 'fever' etc. as names)
    if data["name"] is None:
        name = question.strip()
        # block common symptom/condition words from being names
        blocked_terms = ["fever", "cough", "pain", "rash", "nausea", "headache", "cold", "flu"]
        if len(name) < 2 or not re.match(r"^[A-Za-z\s]+$", name) or name.lower() in blocked_terms:
            return "Please enter your actual name (letters only, not a symptom)."
        data["name"] = name
        session["medical_data"] = data
        session.modified = True
        return "Thank you! What's your age?"

    # Step 2: Get age
    if data["age"] is None:
        if re.match(r"^\d+$", question.strip()):
            age_val = int(question.strip())
            if 1 <= age_val <= 120:
                data["age"] = age_val
                session["medical_data"] = data
                session.modified = True
                return "Got it. Please describe your symptoms clearly (e.g., fever, nausea)."
            else:
                return "Please enter a realistic age (1–120)."
        return "Enter a number."

    # Step 3: Collect symptoms (user typed symptoms; add and return personalized advice + KB lookup)
    std_symptom = standardize_symptom(question)
    existing_std = [standardize_symptom(s) for s in data["symptoms"]]
    if std_symptom not in existing_std:
        data["symptoms"].append(question.strip())
        session["medical_data"] = data
        session.modified = True

    # After collecting at least one symptom, generate personal advice + knowledge lookup
    personal_html = generate_medical_advice(data["name"], data["age"], ", ".join(data["symptoms"]))

    query = ", ".join(data["symptoms"])
    pdf_text, pdf_found = get_pdf_response(query)
    if pdf_found:
        combined = str(personal_html) + "<hr>" + pdf_text + "<br><br><strong>📖 Source:</strong> Local PDF knowledge base"
        return Markup(combined)
    else:
        ext = get_external_response(query)
        combined = str(personal_html) + "<hr>" + "I couldn't find specific details in the PDF.<br><br>" + ext
        return Markup(combined)


# =============== 9. ROUTES ===============
@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname] == pwd:
            session['user'] = uname
            session['chat_history'] = []
            flash("Logged in successfully!", "success")
            return redirect(url_for('chatbot'))
        else:
            error = "Invalid username or password"
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users:
            return "User already exists. <a href='/login'>Login</a>."
        users[uname] = pwd
        flash("Registration successful!", "info")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'user' not in session:
        return redirect(url_for('login'))

    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if not question:
            answer = "Please enter a message."
        else:
            lower_q = question.lower().strip()

            if lower_q in ["hi", "hello", "hey"]:
                answer = "Hi 👋 I'm your Medical Assistant. You can ask about symptoms or medical conditions."

            # If it's an explicit general knowledge query, handle via handle_medical_question (which does PDF-first)
            elif is_general_medical_query(lower_q):
                answer = handle_medical_question(question)

            # If it's a medical/symptom message (or we're already in a medical session), run the personal flow
            elif detect_medical_question(lower_q) or "medical_data" in session:
                session["last_medical_trigger"] = question.strip()
                answer = handle_medical_question(question)

            else:
                # Non-medical queries - but still try PDF first (as requested)
                pdf_text, pdf_found = get_pdf_response(question)
                if pdf_found:
                    answer = Markup(pdf_text + "<br><br><strong>📖 Source:</strong> Local PDF knowledge base")
                else:
                    # fallback to external if PDF didn't have it
                    ext = get_external_response(question)
                    answer = Markup("I couldn't find specific details in the PDF.<br><br>" + ext)

        # store chat history (escape user question; answer may contain markup)
        session['chat_history'].append({
            'question': escape(question),
            'answer': answer
        })
        session.modified = True

    return render_template('chatbot.html', chat_history=session['chat_history'])

@app.route('/clear_chat')
def clear_chat():
    session.pop('chat_history', None)
    session.pop('medical_data', None)
    session.pop('last_medical_trigger', None)
    flash("Chat cleared! 🧹", "info")
    return redirect(url_for('chatbot'))


@app.route('/logout')
def logout():
    username = session.get('user', 'User')
    session.clear()
    flash(f"See you later, {username}! 👋", "info")
    return redirect(url_for('login'))


if __name__ == '__main__':
    print("✅ Medical Chatbot running on http://127.0.0.1:5000")
    app.run(debug=True)

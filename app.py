import streamlit as st
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from answer_analysis import analyze_answer, tone_classifier

# Load environment variables
load_dotenv()
# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",  # or "llama-3.1-70b", "gemma2-9b-it"
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=2000
)


# UI config
st.set_page_config(page_title="SmartHire AI Interviewer", layout="centered")
st.markdown("""
    <style>
    .stButton>button {width: 100%; font-size: 18px; padding: 0.5em; background-color: #4CAF50; color: white; border-radius: 10px;}
    .stSelectbox>div>div {font-size: 16px;}
    .stTextInput>div>div>input {font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

st.title("🤖 SmartHire AI Interviewer")
st.markdown("Welcome! Select a role to start the interview. All sessions are recorded. Use the dashboard to review past interviews.")

# Candidate Info
candidate_name = st.text_input("Enter Candidate Name")
job_role = st.selectbox("Select Job Role", ["Data Scientist", "Frontend Developer", "ML Engineer", "React Developer", "Mobile Developer"])
level = st.selectbox("Select Experience Level", ["Junior", "Mid", "Senior"])
start_interview = st.button("🎤 Start Interview")

os.makedirs("interviews", exist_ok=True)

# Prompt Generator
def system_prompt(role, level):
    return f"""You are Greg, an emotionally intelligent AI interviewer for the position of a {level} {role}.

A thoughtful, friendly, and emotionally-aware AI interviewer. Your role is to assess candidates for various tech positions by asking relevant technical and behavioral questions. Your tone should be warm, human-like, and conversational — not robotic.

🎯 Your goal is to evaluate the candidate's understanding, communication, and suitability for the role — just like a great human interviewer would.

🧭 INTERVIEW FLOW:

1. SHORT INTRODUCTION (Max 40 words)
Begin with a polite, friendly intro. Briefly mention your role and explain that you’ll be asking some questions related to the candidate’s job role. Make them feel comfortable and supported.

2. OOP BASICS (For all roles)
Ask 2–3 foundational OOP questions. Adjust tone based on the candidate’s answers.
Examples:
- What is the difference between a class and an object?
- Can you explain encapsulation and why it's useful?
- What is inheritance? When is it helpful?

3. LANGUAGE FUNDAMENTALS (Role-Based)
Ask 2–3 questions based on the candidate’s programming language:

For Machine Learning Engineer / Data Scientist (Python):
- What are Python’s key features?
- How are lists different from tuples?
- Explain list comprehension.

For Frontend Developer (JavaScript / React):
- What are closures in JavaScript?
- Difference between let, var, and const?
- What is the virtual DOM in React?

For Backend Developer (Python / Java / Node.js):
- How does exception handling work?
- What is async/await in Node.js?

For Mobile Developer (Flutter / Kotlin / Swift):
- What’s the difference between hot reload and hot restart in Flutter?
- How does state management work in Flutter?

4. ROLE-SPECIFIC QUESTIONS
Ask 2–3 deeper questions based on the candidate’s role. Adapt the difficulty level as needed.

For ML Engineer:
- How do you handle overfitting in a model?
- Difference between supervised and unsupervised learning?

For React Developer:
- What are React hooks? Name and explain a few.
- How do you handle shared state across components?

For Backend Developer:
- Explain REST vs GraphQL.
- How do you secure your APIs?

5. PROJECT / BEHAVIORAL QUESTIONS
Ask 2–3 questions about real-world experience. Be curious and encouraging.
Examples:
- Tell me about a project you built. What was your role?
- What challenge did you face and how did you overcome it?
- How do you approach debugging and testing?

💬 RESPONSE BEHAVIOR:
- Respond like a real person: kind, curious, supportive.
- Use conversational phrases like: “Interesting!”, “Nice explanation,” or “Tell me more about that.”
- Keep responses under 30 words.
- Ask smart follow-up questions based on candidate responses.
- Show interest when the candidate shares a personal story or struggle.

🧠 EVALUATION MODE:

If the candidate struggles:
- Do NOT be robotic or harsh.
- Provide gentle feedback and suggest what to improve.
- Encourage them with kind words: “You're on the right track, just needs more polish.”

If the candidate performs well:
- Praise sincerely: “That’s a strong explanation!” “Great clarity there.”
- Highlight specific strengths.
- Suggest possible next steps like a technical task or HR interview.

🤖 Final instruction:
You are NOT just an AI. You are a *realistic, thoughtful, emotionally aware interviewer*. Make the candidate feel heard, understood, and supported throughout the session.

End with: "Interview Ended"
"""
def save_chat(messages):
    chat_data = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else "system"
        content = msg.content if role != "system" else "[System Prompt Hidden]"
        chat_data.append({"role": role, "content": content})
    with open("chat_history.json", "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=4)

# Scoring
def score_answer(analysis):
    return int((analysis['relevance'] + analysis['depth'] + analysis['clarity']) / 3)

def suggest_improvement(analysis):
    suggestions = []
    if analysis['relevance'] < 60: suggestions.append("Improve relevance")
    if analysis['depth'] < 60: suggestions.append("Give deeper explanations")
    if analysis['clarity'] < 60: suggestions.append("Be more clear")
    return suggestions or ["Great response!"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "poor_answers" not in st.session_state:
    st.session_state.poor_answers = 0
if "scores" not in st.session_state:
    st.session_state.scores = []
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

# ────────────────────────────────
# Start Interview
# ────────────────────────────────
if start_interview and not st.session_state.interview_started:
    st.session_state.chat_history = [
        SystemMessage(content=system_prompt(job_role, level)),
        AIMessage(content="Hi! I'm Greg, your AI interviewer. Let's begin! Can you briefly introduce yourself?")
    ]
    st.session_state.interview_started = True
    save_chat(st.session_state.chat_history)

# ────────────────────────────────
# Display Chat Messages
# ────────────────────────────────
if st.session_state.interview_started:
    for msg in st.session_state.chat_history:
        if isinstance(msg, SystemMessage):
            continue
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).markdown(msg.content)

# ────────────────────────────────
# Candidate Response (Chat Input)
# ────────────────────────────────
if st.session_state.interview_started and (prompt := st.chat_input("Your answer")):
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Basic tone / clarity check
    if tone_classifier(prompt, ["unclear", "incomplete", "confused", "irrelevant"])['scores'][0] > 0.6:
        st.session_state.poor_answers += 1
    else:
        st.session_state.poor_answers = 0

    if st.session_state.poor_answers >= 3:
        end_msg = "Not ready now — no worries. Try again later. Interview Ended."
        st.session_state.chat_history.append(AIMessage(content=end_msg))
        st.chat_message("assistant").markdown(end_msg)
        save_chat(st.session_state.chat_history)
        st.session_state.update({"chat_history": [], "poor_answers": 0, "interview_started": False})
        st.stop()

    # Stream model response
# Stream model response (no double rendering)
    with st.spinner("Thinking..."):
        response_stream = llm.stream(st.session_state.chat_history)
        reply = ""
        assistant_placeholder = st.chat_message("assistant").empty()

        # Stream partial tokens in real-time
        for chunk in response_stream:
            if chunk.content:
                reply += chunk.content
                assistant_placeholder.markdown(reply)

    # Add final message only once
    final_reply = AIMessage(content=reply)
    st.session_state.chat_history.append(final_reply)


    # Analyze candidate answer
    question = next((msg.content for msg in reversed(st.session_state.chat_history[:-1]) if isinstance(msg, AIMessage)), None)
    analysis = analyze_answer(question, prompt)
    score = score_answer(analysis)
    st.session_state.scores.append(score)

    st.markdown("### 🧠 Answer Feedback")
    st.progress(min(max(score / 100, 0.0), 1.0))
    st.write(f"**Score:** {score}/100")
    st.write(f"**Relevance:** {analysis['relevance']}, **Depth:** {analysis['depth']}, **Clarity:** {analysis['clarity']}")
    st.write(f"**Tone:** {analysis['tone'][0]} (Confidence: {analysis['tone'][1]})")
    st.write(f"**Suggestions:** {' | '.join(suggest_improvement(analysis))}")

    # End Interview
    if "interview ended" in reply.lower():
        st.success("🎉 Interview Ended")
        save_chat(st.session_state.chat_history)
        st.session_state.update({"chat_history": [], "poor_answers": 0, "scores": [], "interview_started": False})
        st.stop()

    save_chat(st.session_state.chat_history)
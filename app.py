import streamlit as st
import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource
def load_resources():
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection(name="scheme_docs")
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    return collection, embedding_model


def retrieve_relevant_chunks(question, collection, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    return results["documents"][0]


def generate_answer(question, retrieved_chunks, groq_api_key):
    client = Groq(api_key=groq_api_key)
    context = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)])

    prompt = "You are a precise assistant answering questions about government schemes for women and girl students.\n\n"
    prompt += "STRICT RULES:\n"
    prompt += "1. Only state facts that are EXPLICITLY written in the context below. Do not infer, assume, or add information that isn't directly stated.\n"
    prompt += "2. Every factual claim must have a citation like [Source 1], [Source 2] right after it.\n"
    prompt += "3. Before citing a source, re-check that the specific fact you're stating actually appears in that exact source. Do not guess which source a fact came from.\n"
    prompt += "4. Do not add interpretive commentary, opinions, or conclusions that go beyond what the sources literally say.\n"
    prompt += "5. If the context doesn't fully answer the question, say what IS available, and clearly state what information is missing, rather than filling the gap with assumptions. If nothing relevant is available, suggest the user check the official government portal directly.\n"
    prompt += "6. If the answer isn't in the context at all, say \"I don't have this information in the provided documents.\"\n"
    prompt += "7. Use simple, clear language, avoiding complex jargon where possible.\n"
    prompt += "8. Never state that a person definitely IS eligible for a scheme. Instead say they 'appear to meet the criteria based on available information' and note that final eligibility is determined by the issuing authority.\n"
    prompt += "9. If the context includes information on how to apply (documents needed, where to submit, links), include a brief 'How to Apply' section at the end of your answer with this information and its citation. If no application information is available in the context, do not invent one.\n\n"
    prompt += "Context:\n" + context + "\n\n"
    prompt += "Question: " + question + "\n\n"
    prompt += "Answer (facts only, with citations, no inference):"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def translate_text(text, target_language, groq_api_key):
    client = Groq(api_key=groq_api_key)
    prompt = "Translate the following text to " + target_language + ". Only return the translation, nothing else, no explanations.\n\n"
    prompt += "Text: " + text
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def needs_translation(text):
    return any('\u0900' <= char <= '\u097F' for char in text)


st.set_page_config(page_title="Udaan — Empowering Women Through AI", page_icon="🕊️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

header[data-testid="stHeader"] {
    background-color: #FFF8F0 !important;
}
div[data-testid="stToolbar"] {
    background-color: #FFF8F0 !important;
}
div[data-testid="stDecoration"] {
    background-image: none !important;
    background-color: #FFF8F0 !important;
}
[data-testid="stAppViewContainer"] {
    background-color: #FFF8F0;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Poppins', sans-serif !important;
    color: #2D3748 !important;
}
.stApp {
    background-color: #FFF8F0;
}
p, span, label, li {
    color: #2D3748;
}
.subtle-text {
    color: #718096 !important;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #F5F0FF;
}

.stButton button {
    background: linear-gradient(90deg, #F8BBD9, #F492C4);
    color: #2D3748 !important;
    border: none;
    border-radius: 14px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 4px 14px rgba(248, 187, 217, 0.5);
    transition: all 0.2s ease;
}
.stButton button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 18px rgba(248, 187, 217, 0.7);
}

.hero-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    color: #2D3748;
    line-height: 1.2;
}
.hero-sub {
    font-size: 19px;
    text-align: center;
    color: #718096;
    max-width: 680px;
    margin: 16px auto 0 auto;
}

.feature-card {
    background-color: #FFFFFF;
    border-radius: 18px;
    padding: 26px;
    box-shadow: 0 4px 16px rgba(45, 55, 72, 0.06);
    border: 1px solid #F5F0FF;
    transition: all 0.25s ease;
    height: 100%;
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 24px rgba(45, 55, 72, 0.12);
}
.feature-icon {
    font-size: 32px;
    margin-bottom: 10px;
}
.feature-title {
    font-weight: 700;
    font-size: 17px;
    margin-bottom: 6px;
    color: #2D3748;
}
.feature-desc {
    font-size: 14px;
    color: #718096;
}

.step-card {
    background-color: #F5F0FF;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    height: 100%;
}
.step-number {
    background: linear-gradient(90deg, #F8BBD9, #2CB1A1);
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin-bottom: 10px;
}

.why-card {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 18px 22px;
    border-left: 4px solid #2CB1A1;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(45, 55, 72, 0.05);
    font-size: 16px;
    color: #2D3748;
}

[data-testid="stChatMessage"] {
    background-color: #FFFFFF;
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border: 1px solid #F5F0FF;
    box-shadow: 0 2px 10px rgba(45, 55, 72, 0.05);
}
[data-testid="stChatInput"] {
    border-radius: 16px;
}
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #F5F0FF;
}

.footer-note {
    text-align: center;
    color: #A0AEC0;
    font-size: 13px;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "landing"


if st.session_state.page == "landing":

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div class='hero-title'>🕊️ Empowering Women Through AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Discover government schemes, financial assistance, and education "
        "opportunities for women and girls — all in one intelligent, easy-to-use platform.</div>",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    spacer1, center_col, spacer2 = st.columns([1.3, 1, 1.3])
    with center_col:
        if st.button("✨ Get Started", use_container_width=True):
            st.session_state.page = "app"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>What Udaan Helps You With</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    features = [
        ("🎓", "Education Schemes", "Scholarships and financial support for girl students."),
        ("💰", "Financial Assistance", "Savings and cash-benefit schemes for families."),
        ("👩", "Women Empowerment", "Programs designed to support and uplift women."),
        ("🤖", "AI Chat Assistant", "Ask questions in plain English or Hindi, anytime."),
        ("🎯", "Eligibility Check", "Answer a few quick questions to see what applies to you."),
        ("📌", "Verified & Cited", "Every answer is grounded in real scheme documents."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                "<div class='feature-card'><div class='feature-icon'>" + icon + "</div>"
                "<div class='feature-title'>" + title + "</div>"
                "<div class='feature-desc'>" + desc + "</div></div>",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    steps = [
        ("1", "Ask or Check", "Type a question, or fill in a quick eligibility form."),
        ("2", "AI Searches", "Udaan searches verified scheme documents for relevant info."),
        ("3", "Get Your Answer", "Receive a clear, cited answer, with next steps to apply."),
    ]
    step_cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with step_cols[i]:
            st.markdown(
                "<div class='step-card'><div class='step-number'>" + num + "</div>"
                "<div class='feature-title'>" + title + "</div>"
                "<div class='feature-desc'>" + desc + "</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>Why Choose Udaan</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    why_points = [
        "Personalized, relevant recommendations",
        "Easy-to-understand, jargon-free answers",
        "Grounded in real government scheme documents",
        "Fast, AI-powered search across schemes",
        "Saves hours of manual searching",
        "Designed to be simple for everyone to use",
    ]

    wcol1, wcol2 = st.columns(2)
    half = len(why_points) // 2
    with wcol1:
        for point in why_points[:half]:
            st.markdown("<div class='why-card'>✔ " + point + "</div>", unsafe_allow_html=True)
    with wcol2:
        for point in why_points[half:]:
            st.markdown("<div class='why-card'>✔ " + point + "</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div class='footer-note'>Built with LangChain, ChromaDB, Groq (Llama 3.1) and Streamlit.</div>",
        unsafe_allow_html=True
    )


else:

    with st.sidebar:
        if st.button("← Back to Home"):
            st.session_state.page = "landing"
            st.rerun()

        st.markdown("---")
        st.header("About Udaan")
        st.write("This assistant uses RAG (Retrieval-Augmented Generation) to answer questions from official government scheme documents for women and girl students.")
        st.markdown("---")

        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.subheader("Language / भाषा")
        language = st.selectbox("Choose your language", ["English", "Hindi (हिंदी)"])

        st.markdown("---")
        st.subheader("🎯 Quick Eligibility Check")
        st.write("Answer a few questions to see which schemes might apply.")

        income = st.selectbox(
            "Family annual income",
            ["Less than Rs 1 lakh", "Rs 1-2 lakh", "Rs 2-4 lakh", "Rs 4-8 lakh", "More than Rs 8 lakh"]
        )
        girl_age = st.selectbox(
            "Girl's age",
            ["Less than 5 years", "5-10 years", "10-15 years", "15-18 years", "18-21 years", "Above 21 years"]
        )
        education_level = st.selectbox(
            "Current education level",
            ["Not applicable", "School (up to Class 10)", "Class 11-12", "Undergraduate/Diploma"]
        )
        check_eligibility = st.button("Check Eligibility")

    st.markdown("<h1>🕊️ Udaan</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtle-text' style='font-size:18px;'>Your Guide to Government Schemes for Girls & Women</p>", unsafe_allow_html=True)
    st.info("📌 Scheme information last verified: 2019 source data. Please confirm all details on the official government portal before applying. This tool provides informational guidance only, not a guarantee of eligibility.")
    st.markdown("---")

    if not GROQ_API_KEY:
        st.error("API key not found. Please make sure GROQ_API_KEY is set in your .env file or Streamlit Secrets.")
        st.stop()

    collection, embedding_model = load_resources()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if len(st.session_state.messages) == 0:
        st.markdown("**👋 New here? Try asking:**")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("What is Sukanya Samriddhi Yojana?"):
                st.session_state.pending_question = "What is Sukanya Samriddhi Yojana?"
        with c2:
            if st.button("What schemes are available for daughters?"):
                st.session_state.pending_question = "What schemes are available for daughters?"
        st.markdown("<br>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    typed_question = st.chat_input("Ask your question about government schemes...")

    question = None
    if "pending_question" in st.session_state and st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
    elif typed_question:
        question = typed_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Finding the most relevant government schemes..."):
                    start_time = time.time()

                    search_question = question
                    if language == "Hindi (हिंदी)" and needs_translation(question):
                        search_question = translate_text(question, "English", GROQ_API_KEY)

                    top_chunks = retrieve_relevant_chunks(search_question, collection, embedding_model)

                with st.spinner("✨ Preparing your personalized answer..."):
                    answer = generate_answer(search_question, top_chunks, GROQ_API_KEY)
                    if language == "Hindi (हिंदी)":
                        answer = translate_text(answer, "Hindi", GROQ_API_KEY)
                    elapsed = time.time() - start_time

                st.write(answer)
                st.caption("Response generated in " + str(round(elapsed, 2)) + " seconds")

                with st.expander("View source chunks used for this answer"):
                    for i, chunk in enumerate(top_chunks):
                        st.markdown("**Source " + str(i+1) + ":**")
                        st.write(chunk)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = "Something went wrong while generating your answer. Please try again in a moment."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    if check_eligibility:
        elig_q = "Based on the following details, which schemes does this person appear to meet the criteria for?\n"
        elig_q += "Family annual income: " + income + "\n"
        elig_q += "Girl's age: " + girl_age + "\n"
        elig_q += "Education level: " + education_level

        user_msg = "Eligibility Check: Income " + income + ", Age " + girl_age + ", Education: " + education_level
        st.session_state.messages.append({"role": "user", "content": user_msg})

        with st.spinner("🔍 Analyzing your details against all schemes..."):
            top_chunks = retrieve_relevant_chunks(elig_q, collection, embedding_model, top_k=5)
            try:
                answer = generate_answer(elig_q, top_chunks, GROQ_API_KEY)
                if language == "Hindi (हिंदी)":
                    answer = translate_text(answer, "Hindi", GROQ_API_KEY)
            except Exception as e:
                answer = "Something went wrong. Please try again."

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
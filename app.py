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


st.set_page_config(page_title="Udaan", page_icon="🕊️", layout="centered")

custom_css = """
<style>
.stApp {
    max-width: 900px;
    margin: 0 auto;
    background: linear-gradient(180deg, #1a1025 0%, #0e0a14 100%);
}
h1 {
    background: linear-gradient(90deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}
[data-testid="stChatMessage"] {
    background-color: #1e1530;
    border-radius: 16px;
    padding: 12px 18px;
    margin-bottom: 10px;
    border: 1px solid #2d1f47;
}
[data-testid="stSidebar"] {
    background-color: #150d22;
    border-right: 1px solid #2d1f47;
}
.stButton button {
    background: linear-gradient(90deg, #a855f7, #ec4899);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 600;
}
[data-testid="stChatInput"] {
    border-radius: 14px;
}
[data-testid="stExpander"] {
    background-color: #1a1128;
    border-radius: 10px;
    border: 1px solid #2d1f47;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


with st.sidebar:
    st.header("About Udaan")
    st.write("This assistant uses RAG (Retrieval-Augmented Generation) to answer questions from official government scheme documents for women and girl students.")
    st.write("Tech stack: LangChain, ChromaDB, Groq (Llama 3.1), Hugging Face embeddings, Streamlit")
    st.markdown("---")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("Language / भाषा")
    language = st.selectbox("Choose your language", ["English", "Hindi (हिंदी)"])

    st.markdown("---")
    st.subheader("Quick Eligibility Check")
    st.write("Fill in details to check which schemes might apply.")

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


st.title("🕊️ Udaan")
st.markdown("#### Your Guide to Government Schemes for Girls & Women")
st.write("Ask me about scholarships and welfare schemes for women and girl students in India. Let's help you soar!")
st.info("📌 Scheme information last verified: 2019 source data. Amounts and eligibility criteria may have changed — please confirm all details on the official government portal before applying. This tool provides informational guidance only, not a guarantee of eligibility.")
st.markdown("---")

if not GROQ_API_KEY:
    st.error("API key not found. Please make sure your .env file is set up correctly with GROQ_API_KEY.")
    st.stop()

collection, embedding_model = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask your question about government schemes...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and generating answer..."):
                start_time = time.time()

                search_question = question
                if language == "Hindi (हिंदी)" and needs_translation(question):
                    search_question = translate_text(question, "English", GROQ_API_KEY)

                top_chunks = retrieve_relevant_chunks(search_question, collection, embedding_model)
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

    with st.spinner("Checking eligibility across all schemes..."):
        top_chunks = retrieve_relevant_chunks(elig_q, collection, embedding_model, top_k=5)
        try:
            answer = generate_answer(elig_q, top_chunks, GROQ_API_KEY)
            if language == "Hindi (हिंदी)":
                answer = translate_text(answer, "Hindi", GROQ_API_KEY)
        except Exception as e:
            answer = "Something went wrong. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_text(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text


def split_text(text, chunk_size=300, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    return chunks


def generate_embeddings(chunks):
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embeddings = model.encode(chunks)
    return embeddings


def store_in_chromadb(chunks, embeddings):
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(name="scheme_docs")

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks
    )
    return collection


def retrieve_relevant_chunks(question, collection, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    return results["documents"][0]


def generate_answer(question, retrieved_chunks, groq_api_key):
    client = Groq(api_key=groq_api_key)
    context = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)])

    prompt = "Answer the question using ONLY the context below. If the answer isn't in the context, say \"I don't have this information in the provided documents.\"\n\n"
    prompt += "Context:\n" + context + "\n\n"
    prompt += "Question: " + question + "\n\n"
    prompt += "Answer:"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    file_path = "scheme_docs/schemes_daughters.txt"

    print("Loading file...")
    raw_text = load_text(file_path)
    print(f"Loaded {len(raw_text)} characters.\n")

    print("Splitting into chunks...")
    chunks = split_text(raw_text)
    print(f"Created {len(chunks)} chunks.\n")

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print(f"Created {len(embeddings)} embeddings.\n")

    print("Storing in ChromaDB...")
    collection = store_in_chromadb(chunks, embeddings)
    print(f"Stored {collection.count()} items in ChromaDB.\n")

    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    test_question = "What is Sukanya Samriddhi Yojana?"
    print(f"Test Question: {test_question}\n")

    top_chunks = retrieve_relevant_chunks(test_question, collection, embedding_model)

    print("Generating answer using Groq...\n")
    answer = generate_answer(test_question, top_chunks, GROQ_API_KEY)

    print("----- Final Answer -----")
    print(answer)
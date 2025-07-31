import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# LangChain document loaders and FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFPlumberLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Set Hugging Face cache path
os.environ["HF_HOME"] = "D:\\huggingface_cache"

# --- Step 1: Load documents ---
def load_documents():
    print("📄 Loading documents from 'docs/' folder...")
    loaders = [
        DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PDFPlumberLoader),
        DirectoryLoader("docs", glob="**/*.docx", loader_cls=Docx2txtLoader),
    ]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    print(f"✅ Loaded {len(docs)} documents.")
    return docs

# --- Step 2: Create or load FAISS vector store ---
def create_or_load_vector_store(docs, db_path="faiss_index"):
    print("🔍 Creating or loading FAISS vector store...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(db_path):
        print("📦 Loading existing FAISS index...")
        try:
            db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            print("⚠️ Failed to load FAISS index, rebuilding:", e)
            db = FAISS.from_documents(docs, embedding_model)
            db.save_local(db_path)
            print("✅ FAISS index rebuilt and saved.")
    else:
        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(db_path)
        print("✅ FAISS index saved.")
    return db

# --- Step 3: Load T5 model ---
def load_llm(model_id="google/flan-t5-base"):
    print(f"🤖 Loading model '{model_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
    print(f"✅ Model ready (device: {'cuda' if device == 0 else 'cpu'}).")
    return pipe, tokenizer

# --- Step 4: Ask question with context ---
def ask_question(db, llm_pipe, tokenizer, query, max_context_tokens=512, max_answer_tokens=256):
    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Truncate context if too long
    context_tokens = tokenizer.encode(context, truncation=True, max_length=max_context_tokens)
    context = tokenizer.decode(context_tokens, skip_special_tokens=True)

    prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion:\n{query}"
    result = llm_pipe(prompt, max_new_tokens=max_answer_tokens)[0]["generated_text"]
    return result.strip()

# --- Main ---
if __name__ == "__main__":
    try:
        documents = load_documents()
        db = create_or_load_vector_store(documents)

        model_id = "google/flan-t5-base"  # Use a better open model
        llm_pipeline, tokenizer = load_llm(model_id=model_id)

        while True:
            query = input("\n❓ Ask a question (or type 'exit'): ").strip()
            if query.lower() == "exit":
                break
            answer = ask_question(db, llm_pipeline, tokenizer, query)
            print("\n💬 Answer:\n", answer)

    except Exception as e:
        print("❌ Error:", str(e))

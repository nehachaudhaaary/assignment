import os
from typing import List
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import faiss

# Step 1: Data Preparation
def prepare_dataset(data: List[dict], model_name: str) -> FAISS:
    """
    Prepares the dataset by encoding text into embeddings and stores them in a FAISS index.
    :param data: A list of dictionaries with 'question' and 'context'.
    :param model_name: Model name for embedding generation.
    :return: FAISS index for retrieval.
    """
    print("Encoding dataset...")
    embedding_model = SentenceTransformer(model_name)
    texts = [f"Q: {item['question']} C: {item['context']}" for item in data]
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)

    print("Building FAISS index...")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings.cpu().numpy())

    print("Creating vector store...")
    metadata = [{'question': item['question'], 'context': item['context']} for item in data]
    vectorstore = FAISS(embedding_model, faiss_index, metadata)
    return vectorstore

# Step 2: Retrieval Augmented Generation (RAG) Setup
def setup_rag(vectorstore: FAISS, llm_model_name: str, api_key: str) -> RetrievalQA:
    """
    Sets up the RAG pipeline.
    :param vectorstore: FAISS vector store containing encoded data.
    :param llm_model_name: The name of the language model for generation.
    :param api_key: API key for LLM provider.
    :return: A RetrievalQA pipeline.
    """
    print("Setting up RetrievalQA pipeline...")
    llm = HuggingFaceHub(model_name=llm_model_name, api_key=api_key)
    retrieval_qa = RetrievalQA(retriever=vectorstore.as_retriever(), llm=llm)
    return retrieval_qa

# Step 3: Fine-Tuning Dataset Preparation
def prepare_fine_tune_dataset(data: List[dict], save_path: str):
    """
    Prepares a fine-tuning dataset from the provided QA data.
    :param data: A list of dictionaries with 'question', 'context', and 'answer'.
    :param save_path: Path to save the dataset in JSON format.
    """
    print("Preparing fine-tuning dataset...")
    fine_tune_data = [
        {"input_text": f"Q: {item['question']} C: {item['context']}", "output_text": item['answer']}
        for item in data
    ]
    with open(save_path, 'w') as f:
        json.dump(fine_tune_data, f)
    print(f"Fine-tuning dataset saved to {save_path}")

# Step 4: Main Code Execution
def main():
    # Sample dataset
    dataset = [
        {"question": "What is AI?", "context": "AI stands for Artificial Intelligence.", "answer": "Artificial Intelligence."},
        {"question": "Who developed Python?", "context": "Python was developed by Guido van Rossum.", "answer": "Guido van Rossum."},
    ]

    # Paths and configurations
    embedding_model_name = "all-MiniLM-L6-v2"
    llm_model_name = "google/flan-t5-large"
    api_key = "your-huggingface-api-key"

    # Step 1: Prepare FAISS Vector Store
    vectorstore = prepare_dataset(dataset, embedding_model_name)

    # Step 2: Setup RAG
    retrieval_qa = setup_rag(vectorstore, llm_model_name, api_key)

    # Step 3: Interact with QA Bot
    query = "What does AI stand for?"
    print(f"Query: {query}")
    answer = retrieval_qa.run(query)
    print(f"Answer: {answer}")

    # Step 4: Prepare Fine-Tuning Dataset
    fine_tune_save_path = "fine_tune_data.json"
    prepare_fine_tune_dataset(dataset, fine_tune_save_path)

if __name__ == "__main__":
    main()



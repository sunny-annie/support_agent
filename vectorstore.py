import json
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings


# Загружаем JSON с базой знаний
with open("faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Объединяем вопрос и ответ в один текст для индексирования
documents = [
    Document(page_content=f"Q: {item['question']}\nA: {item['answer']}")
    for item in data
    if "question" in item and "answer" in item
]

# Создаём эмбеддинги и FAISS-векторную базу
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)

vectorstore.save_local("faiss_index")

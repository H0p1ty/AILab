from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
LEN_FUNCTION = len

def _load_and_chunk_documents(file_path: str):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=LEN_FUNCTION,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def _create_vector_store(chunks, persist_directory: str):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vector_store.persist()
    return vector_store

def _k_similarity_search(query: str, vector_store: Chroma, k: int) -> list[Document]:
    _relevant_docs = vector_store.similarity_search(query=query, k=k)
    relevant_docs = []
    for document in _relevant_docs:
        if document not in relevant_docs:
            relevant_docs.append(document)
    return relevant_docs

def _get_context(relevant_docs: list[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in relevant_docs])

def context_from_file(file_path: str, query: str, k: int = 3) -> str:
    chunks = _load_and_chunk_documents(file_path)
    vector_store = _create_vector_store(chunks, PERSIST_DIRECTORY)
    relevant_docs = _k_similarity_search(query, vector_store, 3)
    context = _get_context(relevant_docs)
    return context

def context_from_file_wo_kss(file_path: str) -> str:
    chunks = _load_and_chunk_documents(file_path)
    context = _get_context(chunks)
    return context
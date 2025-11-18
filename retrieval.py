import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

PDF_PATH="data/Jeevanth_Bheeman_.pdf"

CHROMA_PATH="chroma_data"

dotenv.load_dotenv()

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 2. Tokenization / Chunking
# Adjust chunk_size & overlap depending on your use case
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
docs = [
    Document(page_content=chunk.page_content, metadata=chunk.metadata)
    for chunk in text_splitter.split_documents(documents)
]

# 3. Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in Chroma
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

# 5. Persist to disk
vector_db.persist()
print(f"Stored {len(docs)} chunks in Chroma at {CHROMA_PATH}")


"""RAG Chatbot with Ollama and LanceDB"""

from pathlib import Path
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import LanceDB
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
import lancedb

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LANCE_DB_DIR = BASE_DIR / "lancedb"

# Initialize models
embed_model = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:1b", temperature=0.2)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

# Load documents
def load_documents():
    docs = []
    for file in DATA_DIR.rglob("*"):
        if file.suffix == ".pdf":
            docs.extend(PyPDFLoader(str(file)).load_and_split(text_splitter))
        elif file.suffix == ".txt":
            docs.extend(TextLoader(str(file)).load_and_split(text_splitter))
    return docs

# Create or load LanceDB
db = lancedb.connect(str(LANCE_DB_DIR))
table_name = "documents"

if table_name in db.table_names():
    lance_db = LanceDB(connection=db, table_name=table_name, embedding=embed_model)
else:
    docs = load_documents()
    lance_db = LanceDB.from_documents(docs, embed_model, connection=db, table_name=table_name)

retriever = lance_db.as_retriever(search_kwargs={"k": 5})

# Prompt
prompt = PromptTemplate(
    template="""You are CiCi, a friendly AI assistant. Answer using only the context provided.

- Casual chat: 1-2 sentences
- Wh-questions: 3-5 sentences with details
- Other: 1-2 sentences
- Unknown: "I couldn't find that information."

Context: {context}
Question: {question}
Answer:""",
    input_variables=["context", "question"],
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)

# Query function
def ask(query: str):
    result = qa_chain.invoke(query)
    answer = result.get("result", "I don't know").strip()
    print(f"\n📌 {answer}")

# Run
ask("what is kilobot")
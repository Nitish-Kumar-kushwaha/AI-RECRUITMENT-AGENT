import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma


# Load API key
load_dotenv()


# Step 1: Load Resume PDFs
def load_resumes():

    documents = []

    folder_path = "data/resumes"

    for file in os.listdir(folder_path):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            documents.extend(docs)

    return documents


# Step 2: Split Documents into Chunks
def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks


# Step 3: Create Vector Database
def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    vectorstore.persist()

    print("Vector Database Created Successfully")

def get_retriever():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    return retriever


if __name__ == "__main__":

    retriever = get_retriever()

    query = "React developer with frontend skills"

    results = retriever.invoke(query)

    print("\nTop Matching Resumes:\n")

    for i, result in enumerate(results):

        print(f"\nResult {i+1}:\n")

        print(result.page_content[:500])
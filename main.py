from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

#load the pdf
pdf_path = "data/sample.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
# --- Step 2: Split documents into chunks ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Each chunk ~1000 characters
    chunk_overlap=200      # Overlap between chunks to preserve context
)
splits = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(splits)}")

# --- Step 3: Create embeddings using HuggingFace ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# --- Step 4: Build a FAISS vector store ---
vectorstore = FAISS.from_documents(splits, embeddings)
# Step 5 - Retrieval
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# step 6: Initialize your LLM
load_dotenv()  # loads .env variables

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    max_new_tokens=256,
     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.3
)


model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Conversation History:
{history}

Transcript Context:
{context}

Question: {question}

Instructions:
- Answer ONLY based on the provided transcript context.
- If the context is insufficient, respond with "I don't know."
""",
    input_variables=['history', 'context', 'question']
)
# --- Q&A memory storage ---
conversation_history = []
#converts raw data to string
def format_history(history):
    if not history:
        return "No prior conversation."
    return "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])
#function that takes only content from all chunks
def format_docs_with_history(retrieved_docs, history):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text



parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(lambda docs: format_docs_with_history(docs, conversation_history)),
    'history': RunnableLambda(lambda _: format_history(conversation_history)),
    'question': RunnablePassthrough()
})


#parser to convert raw ouput to string
parser = StrOutputParser()
#main rag model
main_chain = parallel_chain | prompt | model | parser
#printing the output

# --- Interactive session with memory ---
print("\n📚 PDF Insight RAG Model Ready! Type 'exit' to quit.\n")

while True:
    query = input("Ask a question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    try:
        # -------------------------
        # Show conversation on demand
        # -------------------------
        if query.lower() in ["show conversation", "conversation", "history"]:
            if conversation_history:
                print("\n💾 Conversation History:")
                for i, qa in enumerate(conversation_history, start=1):
                    print(f"{i}. Q: {qa['question']}\n   A: {qa['answer']}\n")
            else:
                print("\n💾 No conversation history yet.\n")
            continue  # skip normal processing

        # -------------------------
        # Normal PDF question
        # -------------------------
        response = main_chain.invoke(query).strip()  # <- remove .content

        # Store in conversation history
        conversation_history.append({"question": query, "answer": response})

        # Print answer
        print("\n📝 Answer:\n", response, "\n")

    except Exception as e:
        print(f"⚠️ Error: {e}")


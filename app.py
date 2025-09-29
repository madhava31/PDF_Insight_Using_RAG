from langchain_community.chat_models import ChatHuggingFace
from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv() 
# --- Step 1: Create reusable objects ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Step 2: Create runnables for each step ---
load_pdf_runnable = RunnableLambda(lambda pdf_path: PyPDFLoader(pdf_path).load())
split_runnable = RunnableLambda(lambda docs: splitter.split_documents(docs))
embedding_runnable = RunnableLambda(lambda docs: embeddings.embed_documents([doc.page_content for doc in docs]))
vectorstore_runnable = RunnableLambda(lambda docs_embeddings: FAISS.from_documents(docs_embeddings, embeddings))

# --- Step 3: Combine them into a chain ---
pre_chain = load_pdf_runnable | split_runnable | vectorstore_runnable  # embeddings are applied inside FAISS

# --- Step 4: Example usage ---
pdf_path = "data\sample.pdf"
retriever = pre_chain.invoke(pdf_path).as_retriever(search_type="similarity", search_kwargs={"k": 4})

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context and conversation history.
If the context is insufficient, just say you don't know.

Conversation History:
{history}

Transcript Context:
{context}

Question: {question}

Instructions:
- If the new question is about previous questions or answers, use ONLY the conversation history.
- If the question is about the PDF, use the PDF context.
- If you cannot find the answer in either, respond with "I don't know."
""",
    input_variables=['history', 'context', 'question']
)

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
parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser
conversation_history = []

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


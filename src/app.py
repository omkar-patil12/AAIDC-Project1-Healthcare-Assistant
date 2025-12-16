import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load documents from the data directory (.txt files).
    """
    results = []
    data_dir = "data"

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            file_path = os.path.join(data_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                results.append(f.read())

    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    """

    def __init__(self):
        # Initialize LLM
        self.llm = self._initialize_llm()

        # Initialize vector database
        self.vector_db = VectorDB()

        # ✅ RAG Prompt Template (FIXED INDENTATION)
        self.prompt_template = ChatPromptTemplate.from_template(
            """
You are an AI assistant designed to answer questions based strictly on a provided knowledge base.

Use ONLY the information contained in the context below to answer the question.
Do not add, infer, or assume any information beyond the given context.

If the answer cannot be found in the context, respond with:
"I’m sorry, I couldn’t find relevant information about this in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        # ✅ Create chain (FIXED INDENTATION)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError("No valid API key found")

    def add_documents(self, documents: List[str]) -> None:
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        # Retrieve relevant chunks
        docs = self.vector_db.search(input, n_results)

        # Combine chunks into context
        context = "\n\n".join(docs)

        # Call LLM
        answer = self.chain.invoke(
            {"context": context, "question": input}
        )

        return answer


def main():
    try:
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        print("\nLoading documents...")
        docs = load_documents()
        print(f"Loaded {len(docs)} documents")

        assistant.add_documents(docs)

        while True:
            question = input("\nPlease enter your question (type 'quit' to exit): ")
            if question.lower() == "quit":
                break

            result = assistant.invoke(question)
            print("\nAnswer:\n", result)

    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()

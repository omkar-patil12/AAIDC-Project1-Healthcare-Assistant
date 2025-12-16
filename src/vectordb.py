import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Create / load collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    # ------------------------------------------------------------------
    # 1️⃣ TEXT CHUNKING
    # ------------------------------------------------------------------
    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        """
        Split text into smaller chunks (word-based).
        """
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(" ".join(current_chunk)) >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ------------------------------------------------------------------
    # 2️⃣ ADD DOCUMENTS TO VECTOR DB
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[str]) -> None:
        print(f"Processing {len(documents)} documents...")

        all_chunks = []
        all_ids = []

        for doc_index, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            for chunk_index, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"doc_{doc_index}_chunk_{chunk_index}")

        if not all_chunks:
            print("No chunks created. Skipping insertion.")
            return

        # Create embeddings
        embeddings = self.embedding_model.encode(all_chunks).tolist()

        # Add to ChromaDB
        self.collection.add(
            documents=all_chunks,
            embeddings=embeddings,
            ids=all_ids,
        )

        print("Documents added to vector database")

    # ------------------------------------------------------------------
    # 3️⃣ SEARCH VECTOR DB
    # ------------------------------------------------------------------
    def search(self, query: str, n_results: int = 5) -> List[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        documents = results.get("documents", [])

        if documents and len(documents) > 0:
            return documents[0]  # flatten list

        return []

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
import numpy as np

class VectorStore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = None
        self.chunks = []
        self.bm25 = None

        print("successfully loaded")

    def create_embeddings(self, chunks):
        self.chunks = chunks
        documents = []
        tokenized_corpus = []

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk['content'],
                metadata={
                    'page': chunk['page'],
                    'type': chunk['type'],
                    'source': chunk['source'],
                    'chunk_id': i
                }
            )
            documents.append(doc)
            # Tokenize for BM25
            tokenized_corpus.append(chunk['content'].lower().split())

        print("Building FAISS index...")
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"Indices built with {len(documents)} vectors")

    def search(self, query, k=5):
        if self.vectorstore is None:
            print("Vectorstore not created")
            return []
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        formatted_results = []
        for i, (doc, score) in enumerate(results):
            formatted_results.append({
                'chunk': {
                    'content': doc.page_content,
                    'page': doc.metadata['page'],
                    'type': doc.metadata['type'],
                    'source': doc.metadata['source']
                },
                'score': float(score),
                'rank': i + 1
            })

        return formatted_results

    def hybrid_search(self, query, k=5, alpha=0.5):
        """
        Combines BM25 and Vector search results using simple weighted scoring.
        alpha: Weight for vector search (0.0 to 1.0)
        """
        if self.vectorstore is None or self.bm25 is None:
            return self.search(query, k=k)

        # 1. Vector Search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=len(self.chunks))

        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores
        # Vector scores (FAISS L2 distance: lower is better, so we invert or use similarity)
        # Note: some FAISS indices return inner product. Here it's likely L2.
        v_scores = np.array([res[1] for res in vector_results])
        if v_scores.max() != v_scores.min():
            v_scores_norm = (v_scores - v_scores.min()) / (v_scores.max() - v_scores.min())
        else:
            v_scores_norm = v_scores

        # Invert vector scores norm because lower is better for L2
        v_scores_final = 1 - v_scores_norm

        # BM25 scores (higher is better)
        if bm25_scores.max() != bm25_scores.min():
            bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_norm = bm25_scores

        # Combine
        combined_scores = {}

        # Map vector results back to indices
        for i, (doc, _) in enumerate(vector_results):
            idx = doc.metadata['chunk_id']
            combined_scores[idx] = alpha * v_scores_final[i] + (1 - alpha) * bm25_norm[idx]

        # Sort and return top k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        formatted_results = []
        for i, (idx, score) in enumerate(sorted_indices):
            chunk = self.chunks[idx]
            formatted_results.append({
                'chunk': chunk,
                'score': float(score),
                'rank': i + 1
            })

        return formatted_results

    def save(self, filepath='vector_store'):
        if self.vectorstore is None:
            print("No vectorstore to save")
            return
        self.vectorstore.save_local(filepath)

        data = {
            'chunks': self.chunks,
            'bm25': self.bm25
        }
        with open(f"{filepath}_data.pkl", 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath='vector_store'):
        self.vectorstore = FAISS.load_local(
            filepath,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        with open(f"{filepath}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.bm25 = data.get('bm25')

        print(f"Loaded vector store and BM25 index")

if __name__ == "__main__":
    test_chunks = [
        {'content': 'Qatar has strong economic growth', 'page': 1, 'type': 'text', 'source': 'Page 1'},
        {'content': 'Banking sector remains healthy', 'page': 2, 'type': 'text', 'source': 'Page 2'},
        {'content': 'IMF recommendations for fiscal policy', 'page': 3, 'type': 'text', 'source': 'Page 3'}
    ]

    print("Testing LangChain Vector Store...")
    store = VectorStore()
    store.create_embeddings(test_chunks)

    results = store.search("What is Qatar's economic situation?", k=2)
    print(f"\nSearch Results:")
    for result in results:
        print(f"Rank {result['rank']}: {result['chunk']['content'][:50]}... (Score: {result['score']:.3f})")
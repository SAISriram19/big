import time
import pandas as pd
from vector_store import VectorStore
import config
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate():
    print("Starting System Evaluation...")
    store = VectorStore(model_name=config.EMBEDDING_MODEL)
    store.load(config.VECTOR_STORE_PATH)

    test_queries = [
        "What is the projected GDP growth for Qatar?",
        "IMF Executive Board conclusions on Qatar",
        "Risks to the economic outlook",
        "Non-hydrocarbon sector performance",
        "Banking sector stability"
    ]

    results_data = []

    for query in test_queries:
        # Measure Vector Search
        start = time.time()
        v_results = store.search(query, k=5)
        v_latency = time.time() - start

        # Measure Hybrid Search
        start = time.time()
        h_results = store.hybrid_search(query, k=5)
        h_latency = time.time() - start

        results_data.append({
            'query': query,
            'type': 'Vector',
            'latency': v_latency,
            'top_score': v_results[0]['score'] if v_results else 0
        })
        results_data.append({
            'query': query,
            'type': 'Hybrid',
            'latency': h_latency,
            'top_score': h_results[0]['score'] if h_results else 0
        })

    df = pd.DataFrame(results_data)
    print("\n--- Evaluation Results ---")
    print(df.groupby('type')['latency'].describe())

    # Generate a simple plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='query', y='latency', hue='type')
    plt.xticks(rotation=45, ha='right')
    plt.title('Search Latency: Vector vs Hybrid')
    plt.tight_layout()
    plt.savefig('evaluation_latency.png')
    print("\nLatency chart saved to evaluation_latency.png")

    # Qualitative check
    print("\n--- Qualitative Check (Top Result) ---")
    for query in test_queries[:2]:
        print(f"\nQuery: {query}")
        h_res = store.hybrid_search(query, k=1)[0]
        print(f"Hybrid Top (Score {h_res['score']:.3f}): {h_res['chunk']['content'][:150]}...")

if __name__ == "__main__":
    evaluate()

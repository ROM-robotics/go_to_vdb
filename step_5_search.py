import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

class VectorDBWithSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.db_path = "step_4_vector_db"
        self.vectors = []
        self.metadata = {}

    def load(self):
        # Vector တွေကို ပြန်ဖတ်မယ်
        vec_path = os.path.join(self.db_path, "vectors.npy")
        meta_path = os.path.join(self.db_path, "metadata.json")
        
        if os.path.exists(vec_path) and os.path.exists(meta_path):
            self.vectors = list(np.load(vec_path))
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            print(f"Loaded {len(self.metadata)} items from disk.")
        else:
            print("No database found to load.")
    
    def search(self, query_text, k=2):
        # 1. Query ကို Vector ပြောင်းမယ်
        query_vector = self.model.encode(query_text)
        
        # 2. Similarity တွက်မယ် (Cosine Similarity)
        # NumPy သုံးပြီး Vector အားလုံးနဲ့ တစ်ပြိုင်တည်း တွက်ချက်ခြင်း
        # Dot product of query and all stored vectors
        dot_products = np.dot(self.vectors, query_vector)
        
        # Magnitudes (Norms) ကို တွက်မယ်
        norm_query = np.linalg.norm(query_vector)
        norm_vectors = np.linalg.norm(self.vectors, axis=1)
        
        # Cosine Similarity formula: (A . B) / (|A| * |B|)
        similarities = dot_products / (norm_vectors * norm_query)
        
        # 3. Top-K Index တွေကို ရှာမယ် (Sorting)
        # argsort က ငယ်စဉ်ကြီးလိုက်စီတာမို့ -similarities သုံးပြီး အကြီးဆုံးကနေ ယူမယ်
        top_k_indices = np.argsort(-similarities)[:k]
        
        # 4. Result ထုတ်ပေးမယ်
        results = []
        for idx in top_k_indices:
            results.append({
                "id": str(idx),
                "text": self.metadata[str(idx)],
                "score": float(similarities[idx])
            })
        return results

# --- စမ်းသပ်ကြည့်ခြင်း ---
search_engine = VectorDBWithSearch()
search_engine.load()

query = "software development"
top_results = search_engine.search(query, k=1)

print(f"Query: {query}")
for res in top_results:
    print(f"Result: {res['text']} (Score: {res['score']:.4f})")

# ဒီနည်းလမ်းက Database ထဲမှာ Data နည်းနည်းပဲ ရှိသေးရင် ပြဿနာမရှိပေမဲ့၊ 
# Data သိန်းချီရှိလာရင် Vector တစ်ခုချင်းစီကို similarity လိုက်တွက်နေရတာ (Brute Force) ဖြစ်လို့ အရမ်းနှေးသွားပါလိမ့်မယ်။ 
# အဲဒါကို ဖြေရှင်းဖို့ Indexing Algorithms (HNSW, Annoy) တွေကို သုံးကြရတာပါ။
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer

class SimplePersistentDB:
    def __init__(self, db_path="step_4_vector_db"):
        self.db_path = db_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectors = []   # List of arrays
        self.metadata = {}  # { "0": "Apple is a fruit", "1": "..." }
        
        # Folder မရှိရင် ဆောက်မယ်
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def add_data(self, text):
        # 1. Generate ID (လက်ရှိ metadata ရဲ့ အရေအတွက်ကို ID အဖြစ် သုံးမယ်)
        new_id = str(len(self.metadata))
        
        # 2. Convert text to vector
        vector = self.model.encode(text)
        
        # 3. Store in memory
        self.vectors.append(vector)
        self.metadata[new_id] = text
        print(f"Stored with ID {new_id}: {text}")

    def save(self):
        # Vector တွေကို NumPy format နဲ့ သိမ်းမယ်
        vec_array = np.array(self.vectors)
        np.save(os.path.join(self.db_path, "vectors.npy"), vec_array)
        
        # Metadata ကို JSON နဲ့ သိမ်းမယ်
        with open(os.path.join(self.db_path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
        print("Database saved to disk!")

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

# --- စမ်းသပ်ကြည့်ခြင်း ---
db = SimplePersistentDB()

# ၁။ Data အသစ်ထည့်မယ်
db.add_data("Python is a programming language")
db.add_data("The weather is nice today")

# ၂။ Disk ပေါ်မှာ သိမ်းမယ်
db.save()

# ၃။ Load ပြန်လုပ်ကြည့်မယ် (Database အသစ်တစ်ခုအနေနဲ့)
new_db_instance = SimplePersistentDB()
new_db_instance.load()
print("Loaded Metadata:", new_db_instance.metadata)
print("Loaded Vectors Shape:", np.array(new_db_instance.vectors).shape)
print("Vector for ID 0:", new_db_instance.vectors[0])
print("Vector for ID 1:", new_db_instance.vectors[1])
print("Step 4 complete!")
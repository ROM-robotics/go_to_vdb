from sentence_transformers import SentenceTransformer

class SimpleVectorDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectors = []
        self.metadata = []

    def add_data(self, text):
        # စာသားကို Vector ပြောင်းတယ်
        vector = self.model.encode(text)
        # သိမ်းဆည်းတယ်
        self.vectors.append(vector)
        self.metadata.append(text)
        print(f"Added to DB: {text}")

    def get_all(self):
        """Output ပြန်ထုတ်ကြည့်ခြင်း"""
        return list(zip(self.metadata, self.vectors))

# စမ်းသပ်ကြည့်ခြင်း
db = SimpleVectorDB()
db.add_data("Apple is a fruit")
db.add_data("Programming is fun")

print("\nStored Data:")
for text, vec in db.get_all():
    print(f"Text: {text}, Vector: {vec}")
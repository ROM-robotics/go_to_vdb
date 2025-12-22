import numpy as np

class SimpleVectorDB:
    def __init__(self):
        # Vector တွေကို သိမ်းဖို့ နေရာ
        self.vectors = [] 
        # စာသားတွေကို သိမ်းဖို့ နေရာ
        self.metadata = []

    def add_vector(self, vector, text):
        """Input ထည့်ခြင်း"""
        self.vectors.append(vector)
        self.metadata.append(text)
        print(f"Added: {text}")

    def get_all(self):
        """Output ပြန်ထုတ်ကြည့်ခြင်း"""
        return list(zip(self.metadata, self.vectors))

# စမ်းသပ်ကြည့်ခြင်း
db = SimpleVectorDB()

# စိတ်ကူးထဲက Vector လေးတွေ ထည့်မယ် (တကယ်ဆိုရင် Model တစ်ခုခုက လာရမှာပါ)
db.add_vector([0.1, 0.2, 0.3], "Apple")
db.add_vector([0.9, 0.8, 0.7], "Banana")

# သိမ်းထားတာတွေ ပြန်ကြည့်မယ်
print("\nStored Data:")
for text, vec in db.get_all():
    print(f"Text: {text}, Vector: {vec}")
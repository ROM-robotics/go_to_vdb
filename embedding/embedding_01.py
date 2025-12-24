"""
ဤဖိုင်မှာ PyTorch ရဲ့ nn.Embedding ကို အသုံးပြုပြီး
token id များကို vector များ (embedding vectors) အဖြစ်ပြောင်းလဲတဲ့
နည်းလမ်းကို ပြသထားပါတယ်။

အဓိကအဆင့်များ
- vocab_size ကို သတ်မှတ်ပြီး token အရေအတွက်ကို ပြောရှင်းထားသည်
- d_model က token တစ်လုံးချင်းစီကို ကိုယ်စားပြုမယ့် dimension အရေအတွက်
- nn.Embedding(vocab_size, d_model) နဲ့ embedding matrix တစ်ခုဖန်တီးရန်
- token_ids ကို ထည့်ပြီး lookup လုပ်ကာ vectors ထုတ်ယူရန်
- shape (အရွယ်အစား) များကို စစ်ဆေးပြရန်
"""

import torch
import torch.nn as nn

vocab_size = 10  # Vocabulary ထဲမှာ token အမျိုးအစား ၁၀ ခုရှိတယ်
d_model = 4      # Token တစ်လုံးကို ၄-ဒိုင်ခွန်းကျယ် (dimension=4) နဲ့ ကိုယ်စားပြုမယ်

embedding = nn.Embedding(vocab_size, d_model)
# Embedding matrix သည် စစ်မှန်သော သင်ကြားရမည့် parameter များအဖြစ် စိတ်ကျေနပ်မှုမရှိသေးသော
# စတင်တန်ဖိုးများ (random init) ဖြစ်သည်။
# Shape သည် (vocab_size, d_model) ဖြစ်ပြီး row တစ်ခုစီက token တစ်လုံး၏ vector ကို ကိုယ်စားပြုသည်။
print("Embedding:", embedding.weight)
print("Embedding Shape:", embedding.weight.shape)  # (vocab_size, d_model)


token_ids = torch.tensor([1, 3, 5])  # ဥပမာ token id များ ၁၊ ၃၊ ၅ ကို ရွေးချယ်ထားသည်
vectors = embedding(token_ids)       # id တန်ဖိုးများကို အသုံးပြုပြီး row-wise lookup လုပ်ရန်

print(vectors)           # ထွက်လာသော vectors သည် (num_tokens, d_model) အရွယ်အစားရှိမည်
print(vectors.shape)     # ဥပမာ (3, 4) — token ၃ လုံး ၊ dimension ၄

## Attention Heads


Embedding က ထွက်လာတဲ့ semantic meaning တွေပါဝင်တဲ့ word vectors တွေကို contextual meaning ရအောင်လို့ Transformer ရဲ့ attention is all you need ဆိုတဲ့ attention layers ထဲကို ဝင်ရောက်ပါမယ်။ 

Attention Mechanism တွေထဲမှာ၊  
- Self Attention
- Casual Attention
- Cross Attention
- Multi-head attention 

ဆိုပြီး Mechanism တွေပါဝင်ပါတယ်။ 

Attention (Q,K,V) = softmax(QK**T/square root of dk) V

Token တစ်ခုနဲ့တစ်ခု relation ရှိမရှိ မတွက်ခင်မှာ Embedding Layer ထဲက ထွက်လာတဲ့ Embedding Vector + Positional Encoding ပေါင်းထားတဲ့၊ Vector ကို Attention Head ထဲက Weight Matrix တွေနဲ့ Matrix Transformation လုပ်ပါသေးတယ်။ (ဒီ weight matrix တွေဟာ Neural Network နဲ့ Trainable parameters တွေပါ။ ) Query Weight Matrix, Value Weight Matrix နဲ့ Key Weight Matrix ထဲကို pass ပေးပြီး Transformation လုပ်ပြီးပြီဆိုရင် Embedding Vectors တွေဟာ Query Matrix, Value Matrix, Key Matrix စသဖြင့် သုံးခု ရရှိလာမှာပါ။ အာ့တာရလာတာနဲ့ Token တစ်ခုနဲ့တစ်ခု relation ရှိလားမရှိလား စတင်လုပ်ဆောင်မှာပါ။ 

Token တစ်ခုနဲတစ်ခု relation ရှိလားမရှိလား ဆိုတာ ဥပမာ - Sentences မှာ Token နှစ်ခုရှိမယ်ဆိုပါစို့

Token A ဟာ Token B နဲ့ ဘယ်လောက်နီးစပ်သလဲဆိုတာ တိုင်းတာဖို့အတွက်၊ 

Token A မှာရှိတဲ့ Unique ဖြစ်တဲ့ Query, Key, Value Matrix သုံးခုထဲက Query Matrix နဲ့ Token B ရဲ့ Transpose Key Matrix နှစ်ခုကို Matrix Multiplication လုပ်ပါတယ်။ အာ့သလိုပဲ။

Token A ရဲ့ Query Matrix နဲ့ Token A ရဲ့ Key Matrix Transpose
Token A ရဲ့ Query Matrix နဲ့ Token B ရဲ့ Key Matrix Transpose
Token B ရဲ့ Query Matrix နဲ့ Token B ရဲ့ Key Matrix Transpose
Token B ရဲ့ Query Matrix နဲ့ Token A ရဲ့ Key Matrix Transpose 

Matrix Multiplication တွေလုပ်ပြီးတဲ့ အခါ attention score တွေရရှိပါတယ်။ AA, BB, AB, BA. ဆိုပြီး
RAW Attention Score တွေကို Matrix တစ်ခုထဲမှာစုပေါင်းပြီးသိမ်းဆည်းကြရပါတယ်။ 

Token နှစ်ခုရှိရင် Matrix က 2x2 size နဲ့ Score အရေအတွက် ၄ ခု ရမှာဖြစ်ပြီ။ Sentense တစ်ကြောင်းမှာ 5 Tokens ရှိရင် 5x5 Matrix နဲ့ 25 Score ရရှိမှာဖြစ်ပါတယ်။ 

မှတ်ချက်။ ။ Tokens ထဲမှာရှိတဲ့ information တွေကို Query က ဆွဲထုတ်လာပြီး Key ကိုမေးတဲ့သဘောဖြစ်ပါတယ်။ ရလာတဲ့ matrix က RAW Attention Score ပဲဖြစ်ပါတယ်။

ဒါဆိုရင် Attention (Q,K) ရဲ့ equation ထဲက (QK**T) ဆိုတဲ့အပိုင်းကို ရပါပြီ။ softmax ထဲဝင်ဖို့ကျန်ပါတယ်။ Softmax ဟာ lists of numbers တွေကို probabilities တွေပြောင်းပေးတဲ့ method ပါ။
သူ့ကို ထည့်ပေးလိုက်တဲ့ number lists တွေမှာ value တွေဟာ အတော်လေးကွာဟပြီး range or variance ကြီးတယ်ဆိုရင် ဥပမာ- score တွေဟာ 10, -10, 1, 2 နဲ့ရှိတယ်ဆိုပါစို့, 10 က အများဆုံး value ဖြစ်သွားပြီး softmax result ကတော့ 1 နားမှာနေမယ်၊ ဒါပေမယ့် -10 ကတော့ 0 နားမှာနေလိမ့်မယ်။ 

ဒါကို normalized လုပ်ပေးဖို့လိုတဲ့အတွက် square root of dimension key vector နဲ့ divide လုပ်လိုက်ရပါတယ်။ ပြီးမှ softmax ထဲထည့်ပေးလိုက်ပါတယ်။

နောက်ဆုံးမှာ probabilities score တွေပါဝင်တဲ့ သို့မဟုတ် Attention Weight တွေပါဝင်တဲ့ matrix တစ်ခုရရှိပါတယ်။

ရလာတဲ့ Attention Weight matrix နဲ့ ကျန်ရှိနေသေးတဲ့ Value Matrix နဲ့ Matrix Multiplication လုပ်ပါတယ်။ ရလာတဲ့ Matrix ဟာ နောက်ဆုံး representation matrix အနေနဲ့ ရှိမယ့်ကောင်ပါ။ ဒါကို Tokens အားလုံးအတွက် Contexual final Embedding Matrix လို့ခေါ်မှာဖြစ်ပါတယ်။ 

ဒါကို self attention Mechanisms လို့ခေါ်ပါမယ်။ Transforming static embedding into contextual word embeddings.
--- 
## Causal Self-Attention ဆိုတာ

future token တွေကို မကြည့်ဘဲ past + current token တွေကိုပဲ attention ပေးခိုင်းတဲ့ self-attention နည်းလမ်း ဖြစ်ပါတယ်။

ဒီ mechanism ကို GPT, Decoder-only LLMs, Text generation models တွေမှာ မဖြစ်မနေ သုံးရပါတယ်။

ဘာကြောင့် “future token” ကို မကြည့်ခိုင်းရတာလဲ ဆိုရင် Classification (BERT လို model) မျိုးမှာ Sentence တစ်ကြောင်းလုံးကို တစ်ပြိုင်နက်တည်း မြင်ရပြီးတော့ ဒီ sentence က ဘာကို ဆိုလိုတာလဲ” ဆိုတာကို တစ်ခါတည်းတန်းဆုံးဖြတ်ရတာဖြစ်ပါတယ်။ အာ့တော့ future/past ဆိုတာမရှိတော့ပါဘူး။ full context meaning တွေပါဝင်တဲ့ embeddeding matrix ကို probability score matrix ထုတ်ပြီးတော့ classify လုပ်ပစ်ရတာပါ။ ဒါပေမယ့် LLMs Generator model မျိုးတွေမှာတော့ အာ့လိုမရပါဘူး။ 

#### Mask matrix ရဲ့ idea
future positions → ❌ attention မပေး
past + current positions → ✅ attention ပေး

အာ့ကြောင့် masking matrix တစ်ခုဖန်တီးပါတယ်။ အာ့ matrix ရဲ့ size က input sequences size နဲ့အတူတူပါပဲ။ အာ့ matrix မှာ negative infinity တန်ဖိုးတွေပါဝင်တဲ့ vectors တွေကို tokens ကို predicts (attention) မလုပ်စေချင်တဲ့ တန်ဖိုးတွေနေရာမှာ ထားပြီး ပေါင်းထည့်လိုက်ပါတယ်။ 

[ 0    -∞    -∞    -∞ ]
[ 0     0    -∞    -∞ ]
[ 0     0     0    -∞ ]
[ 0     0     0     0 ]

    ဒီ matrix မှာဆိုရင် attention weight (သို့မဟုတ်) probability တန်ဖိုးတွေကို past + current position တွေမှာပဲထည့်ထားပြီး ကျန် position တွေမှာ - infinity ကို အစားထိုးထားပါတယ်။ -∞ → attention blocked

scores = QKᵀ / √d
masked_scores = scores + mask
softmax(-∞) = 0
Attention = softmax(masked_scores) × V

Decoder LLM မှာ
attention output က token အားလုံးအတွက် ထွက်ပေမယ့်
loss / prediction ကို last token တစ်ခုတည်းပဲ သုံးပြီးတွက်ပါတယ်။


ဒီ masking matrix နဲ့ original attention score matrix ကိုပေါင်းလိုက်ပြီး ရလာတဲ့ attention weight Matrix ကို Value Matrix နဲ့ Matrix Multiplication လုပ်လိုက်တယ်ဆိုရင် Final Embedding Matrix ပဲထပ်ရပါတယ်။


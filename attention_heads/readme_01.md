## Attention Heads

ဤမှတ်စုသည် Transformer မော်ဒယ်များထဲရှိ Attention Mechanism ကို မြန်မာလို ချုပ်ချုပ်တိုတိုနဲ့ဖတ်ရလွယ်အောင် ရှင်းလင်းဖော်ပြထားပါတယ်။

### အကျဉ်းချုပ်
- Embedding လေးများက semantic meaning ကို ကိုယ်စားပြုမယ်။
- Attention ကတော့ token တစ်လုံးနှင့်တစ်လုံးနှစ်ခုကြား ဆက်နွယ်မှုကို တွက်ချက်ပြီး contextual meaning ကို တိုးတက်စေမယ်။

### Attention မော်ဒယ်အမျိုးအစားများ
- Self Attention
- Causal Attention (မူရင်း “Casual” မဟုတ်ပါ)
- Cross Attention
- Multi-Head Attention

### သင်္ချာဖော်ပြချက်
Attention ကို အောက်ပါအတိုင်းတွက်ချက်သည် —

$$\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

အလိုက်လိုက်လုပ်ဆောင်ပုံကို အဆင့်လိုက် ဖော်ပြပါသည်။

### အဆင့်လိုက်လုပ်ဆောင်မှု
1) Embedding + Positional Encoding ကို ပေါင်းစပ်ပြီး token vectors ကိုပြင်ဆင်သည်။

2) Linear Projections ဖြင့် $Q, K, V$ တို့ကိုရယူသည်။
	- $Q = X W_Q$, $K = X W_K$, $V = X W_V$
	- $W_Q, W_K, W_V$ များသည် trainable weight matrices ဖြစ်သည်။

3) Attention Scores ကိုတွက်ချက်သည်။
	- $S = QK^\top$ (token-to-token similarity)
	- Scale: $S/\sqrt{d_k}$ (variance ပြေလျော့ရေးအတွက်)

4) Softmax ဖြင့် Attention Weights ရယူသည်။
	- $A = \mathrm{softmax}(S/\sqrt{d_k})$
	- Row တစ်ကြောင်းစီသည် Query token တစ်လုံး၏ distribution ဖြစ်သည်။

5) Value များပေါ်တွင် weighted sum လုပ်ပြီး context-aware representation ကိုရယူသည်။
	- $O = A V$

### Token နှစ်ခု ဥပမာ (Matrix Size ကို သဘောပေါက်ရန်)
- Token A, Token B ဆိုပြီး ၂ လုံးရှိသော် $S = QK^\top$ သည် $2\times2$ matrix ဖြစ်မည် — AA, AB, BA, BB ဆိုပြီး score ၄ ခု ရရှိမည်။
- Sentence တစ်ကြောင်းတွင် tokens $n$ လုံးရှိလျှင် $S$ သည် $n\times n$ matrix ဖြစ်ပြီး score $n^2$ ခု ရမည်။ ဥပမာ $n=5$ ဖြစ်လျှင် $5\times5 = 25$ score။

မှတ်ချက် — Query သည် information ကို “မေးမြန်း/ဆွဲထုတ်” နေပြီး Key သည် “ပါဝင်ဆောင်ရွက်နေသော သော့ချက်” ကဲ့သို့ အဓိပ္ပါယ်ယူဆနိုင်သည်။ $QK^\top$ သည် raw attention scores ဖြစ်ပြီး Softmax ပြီးမှ probabilities (attention weights) ထွက်ပေါ်မည်။

### Causal / Cross / Multi-Head အကြောင်းအခြေခံ
- Causal Attention: အနာဂတ် token များကို ကြည့်မရအောင် mask ထားပြီး left-to-right တန်ဖိုးများသာ သုံးသည်။
- Cross Attention: Encoder outputs ကို Key/Value အဖြစ်ယူပြီး Decoder queries နှင့် ဆက်သွယ်ပေးသည်။
- Multi-Head Attention: Head များစွာဖြင့် (parallel) attention တွက်ပြီး concat → linear projection ပြုလုပ်ကာ richer relations ကို စုစည်းတင်ပြသည်။



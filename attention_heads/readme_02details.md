## Greedy Sampling 

Prompt Generation မှာ RAW Logits ကနေထွက်လာတဲ့ Similarity Tokens တွေကို Probabilities Distrubtion အရ တန်ဖိုး အဖြစ်နိုင်ဆုံးကောင်ကို ဆွဲထုတ်ပါတယ်။

အာ့လိုဆွဲထုတ်တဲ့အတွက် AI Model က repeat ထပ်ပြီးအမြဲတမ်း the same token (with highest probabilites) တန်ဖိုးရှိတဲ့ token ကို ပဲ အဖြေပြန်ပေးနေတာပါ။ အာ့လိုအဖြေပြန်ပေးနေတာကို Greedy Sampling လို့သတ်မှတ်ပါတယ်၊ Model က crative မဖြစ်သလို စိတ်ဝင်စားစရာလည်းမကောင်းပါဘူး။ 

## Temperature Scaling 

Temperature Scaling က ဘယ်အချိန်မှာလုပ်သလဲဆိုရင် RAW Logics ထွက်လာတဲ့ တန်ဖိုးတွေကို Softmax ထဲမဝင်ခင်မှာ Temperature Value တစ်ခုနဲ့ စား (divide) လိုက်တာပဲဖြစ်ပါတယ်။ Temperature Value က 0 ကနေ စပြီး any positive number ဖြစ်ရပါတယ်။ Default အရ 1 ဖြစ်ပါတယ်။


1 ထက်ငယ်ရင် More Confident (or) More Deterministics ( Less Random )
1 ထက်ကြီးရင် More Creative (More Random) - 1 ထက်ကြီးတဲ့အခါ less likey tokens တွေရဲ့ Logits တန်ဖိုးဟာ တိုးလာပါတယ်။

အများအားဖြင့် Temperature Scaling လုပ်တဲ့အချိန်မှာ အသုံးပြုမယ့် Value ကို LLMs Model တွေရဲ့ provider က သတ်မှတ်ပေးတဲ့ value range အတွင်းမှာပဲကစားလို့ရပါမယ်။ ဥပမာ - 0 to 2 
2 ထက်ကြီးသွားတဲ့အခါ တစ်ခါတစ်လေ model က inaccurate ဖြစ်သွားတက်သလို့၊ ထွက်လာတဲ့ prompt တွေကလည်း အဓိပ္ပါယ်မရှိတော့ဘူးဖြစ်သွားတက်ပါတယ်။

## Top K Sampling 

Top K Sampling လုပ်တယ်ဆိုတာ ဘယ်အချိန်မှာလုပ်တာလဲဆိုရင် Raw Logits > Temp Scaling > Softmax (Probabilites Distribution) ထွက်လာတဲ့အချိန်မှာ လုပ်တာပါ။ ဥပမာ- Top K = 5 ဆိုရင် Probabilites Distribution ထဲက အမြင့်ဆုံး token 5 ခုကို အသုံးပြုမယ်လို့ပြောတာဖြစ်ပါတယ်။ Top K Sampling ကိုသုံးလိုက်ခြင်းအားဖြင့် Softmax ထဲက ထွက်လာတဲ့ Probabilites distribution ထဲက  တန်ဖိုးနိမ့်တဲ့ Tokens တွေကို မယူတော့ပဲ Top K က select လုပ်ထားပြီးသား 5 ခုသာ ကျန်ပါတော့မှာဖြစ်ပါတယ်။ အာ့ငါးခုကို  probabilites dist ( 1 ) ရအောင် renormalized  ပြန်လုပ်ပြီးမှ အဖြစ်နိုင်ဆုံးကောင်ကို ရွေးချယ် မှာဖြစ်ပါတယ်။

## Top P (Nucleus) Sampling
Softmax က ထွက်လာတဲ့ Cumulative Sum က 1 ဆိုတာသိပါတယ်။ Top P = 0.9 ဆိုပါစို့၊ ပထမဦးဆုံး softmax ထဲက ထွက်လာတဲ့ကောင်တွေကို probabilities Distributions တန်ဖိုးကြီးစဥ်ငယ်လိုက်စီလိုက်ပါမယ်။ ပြီးရင် ကြီးစဥ်ငယ်လိုက် Token တွေကရဲ့ probabilities တန်ဖိုးတွေကို 0.9 မရမခြင်းပေါင်းသွားမှာဖြစ်ပါတယ်။ 
0.9 Threadshold ရောက်ပြီဆိုတာနဲ့ Token selections တွေကိုရပ်လိုက်ပြီး အာ့ဒီ 0.9 တန်ဖိုးအထိ ပေါင်းနေတဲ့အချိန်တွင်းမှာပါဝင်တဲ့ Tokens တွေကို renomalized ပြန်လုပ် prob dist ပြန်ထုတ်ပြီး အာ့ထဲက ရွေးချယ်တာဖြစ်ပါတယ်။ ဒါဆိုရင် Top P = 0.0 ပဲထားလိုက်မယ်ဆိုရင်  softmax က ထွက်လာတဲ့ probabilities တန်ဖိုးအမြင့်ဆုံးကောင်ကို အရင်တုန်းက Greedy sampling လုပ်ခဲ့တုန်းကလိုပဲ ရွေးချယ်သွားမှာဖြစ်ပါတယ်။  Top P = 1 ထားမယ်ဆိုရင်လည်း 1 ထိရအောင်ပြန်ပေါင်းတာဖြစ်တဲ့အတွက် softmax distribution output အကုန်လုံးပြန်ပါနေမှာပဲဖြစ်ပါတယ်။


### NOTE: Top K က fixed token selection လုပ်မှာဖြစ်ပြီး၊ Top P ကတော့ သူ့ကို သတ်မှတ်ထားတဲ့ Probabilites Threadshold တန်ဖိုးအပေါ်မူတည်ပြီး Cumulative sum လုပ်၊ ရသလောက် Token ကိုယူမှာဖြစ်ပါတယ်။
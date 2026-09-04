# 🏗️ Project 1 — Baseline: Train a Tiny CNN on MNIST

> ⭐ **ادعمنا بنجمة:** إذا أعجبك المشروع أو استفدت منه، لا تنسَ أن تمنحنا ⭐ نجمة — دعمك يصنع فرقاً كبيراً!
> ⭐ **Give us a star:** if you like this project or learned something from it, please give us a ⭐ star — your support means a lot!

> **The Why:** قبل أن ندرس هجمات أمن تعلّم الآلة لا بد من "مريض سليم": موديل صغير مدرب على MNIST نعرف دقته النظيفة تماماً — وبه نقارن كل تجربة هجوم أو دفاع في السلسلة. ستتعلم هنا بناء وتدريب CNN صغير مناسب لـ CPU و Colab المجاني، وقراءة "ثقة الموديل" التي ستستغلها الهجمات لاحقاً.
>
> **The Why (EN):** before studying ML-security attacks we need a healthy "patient": a small CNN trained on MNIST whose clean accuracy we know exactly — the reference point for every attack/defense experiment in this series. You will learn to build and train a CPU-friendly CNN and read the model confidence that attacks later exploit.

<div align="center">

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdrabuha/AI-ML-Security/blob/main/01-baseline-mnist-cnn/baseline_mnist_cnn.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()

`baseline_mnist_cnn.ipynb` &nbsp;·&nbsp; `baseline_mnist_cnn.py` — نفس الكود للطرفية / same code for terminal

</div>

## 🎯 What you will learn / ماذا ستتعلم

- تحميل MNIST وتقسيمه (12k تدريب / 2k اختبار) / loading & splitting MNIST
- معمارية CNN صغيرة ومفهومة (~133k معامل) / a tiny, explainable CNN
- حلقة تدريب + تقييم + حفظ الموديل / train → evaluate → save loop
- قراءة توزيع الاحتمالات (Softmax) و"الثقة" / reading Softmax confidence

## 🧠 Data flow / تدفق البيانات

```text
MNIST (28×28) ──► TinyCNN ──► 10 logits ──► softmax ──► confidence
  60k digits      Conv→Pool→      one per        "I'm 97% sure
  0 ... 9         Conv→Pool→      digit 0..9     this is a 7"
                  Dense→Dense
```

## 🚀 Quickstart

```bash
# Colab (موصى به / recommended): اضغط زر Open in Colab بالأعلى
pip install -q torch torchvision matplotlib
python baseline_mnist_cnn.py
```

**What you should see / ماذا سترى:** دقة اختبار نظيفة ≈ **97–98%**، مع طباعة ثقة الموديل على 5 صور.

## 🔬 Visual experiments / تجارب بصرية

1. **شبكة 6 صور** من MNIST مع تسمياتها الحقيقية.
2. **عدّاد المعاملات** (~133k) — لماذا هو صغير؟ لأن الصور 28×28.
3. **ثقة الموديل** على 5 صور: true vs predicted vs confidence.

## 🛠️ Troubleshooting

| المشكلة / Problem | الحل / Fix |
|---|---|
| تحميل MNIST بطيء أول مرة | طبيعي (~10 MB) — يتم مرة واحدة |
| الدقة أقل من 95% | زد `epochs` أو استخدم كل الـ 60k عينة |
| نفاد الذاكرة | قلل `BATCH` إلى 32 |

## 📁 Structure

```text
01-baseline-mnist-cnn/
├── README.md
├── baseline_mnist_cnn.ipynb
└── baseline_mnist_cnn.py
```

## ➡️ Next

**[Project 2 — Evasion Attacks](../02-evasion-attacks/)**: أضف ضوضاء FGSM خبيثة لهذا الموديل السليم وشاهد الدقة تنهار بينما الصورة لا تتغير للعين. ⚔️

---

## 💛 Support the Author / دعم المؤلف

> **العربية:** إذا أردت دعمي، فنشكرك ونقدّر لك ذلك بصدق. ❤️ يمكنك إرسال الدعم عبر العناوين التالية.
>
> **English:** If you want to support me, we thank and appreciate that. ❤️ You can send a donation using the addresses below.

| العملة / Coin | العنوان / Address |
|---|---|
| Bitcoin (BTC) | _يُضاف لاحقاً · to be added_ |
| Ethereum (ETH) | _يُضاف لاحقاً · to be added_ |
| Solana (SOL) | _يُضاف لاحقاً · to be added_ |
| USDT | _يُضاف لاحقاً · to be added_ |
| USDC | _يُضاف لاحقاً · to be added_ |

---

## ©️ الحقوق والترخيص · Copyright & License

**المؤلف / Author:** عبدربه العتيبي (Abdrabuh Alotaibi) · [abdrabuha@outlook.com](mailto:abdrabuha@outlook.com)

**العربية:** © 2026 عبدربه العتيبي — هذا المشروع مُرخَّص بموجب [رخصة MIT](../LICENSE)؛ يمكنك استخدام الأكواد وتعديلها وتعلُّمها ومشاركتها بحرية مع الإبقاء على حقوق المؤلف.

**English:** © 2026 Abdrabuh Alotaibi — this project is licensed under the [MIT License](../LICENSE); you may freely use, modify, learn from, and share the code, provided you keep this copyright notice.

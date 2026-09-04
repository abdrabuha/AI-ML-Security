<div align="center">

# 🛡️ AI-ML-Security Learning Series

> ⭐ **ادعمنا بنجمة:** إذا أعجبك المشروع أو استفدت منه، لا تنسَ أن تمنحنا ⭐ نجمة — دعمك يصنع فرقاً كبيراً!
> ⭐ **Give us a star:** if you like this project or learned something from it, please give us a ⭐ star — your support means a lot!
>
> 🤝 **قاعدة المسابقات:** إذا شاركت في مسابقة أو هاكاثون باستخدام هذا المشروع أو محتواه، **يجب عليك إضافتي كعضوٍ في فريقك** — إضافتي شرط إلزامي!
> 🤝 **Competitions rule:** if you enter a competition or hackathon using this project or its content, **you must add me to your team** — my inclusion is mandatory!

### Practical, hands-on notebooks about Adversarial Machine Learning — attacks & defenses on small models.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Colab](https://img.shields.io/badge/Ready%20for-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

**4 projects · small models (MNIST CNN) · free Colab CPU/T4 · from a clean baseline to robust models**

</div>

---

## 🌍 مرحباً / Welcome

> **بالعربية:** سلسلة مشاريع تعليمية عملية تشرح **أمن تعلّم الآلة** (Adversarial Machine Learning)
> بأسلوب "تعلّم بالممارسة": ندرّب موديل صغير على MNIST، ثم نهاجمه بهجمات التهرب (Evasion:
> FGSM و PGD)، ونتعلّم هجمات التسميم والأبواب الخلفية (Poisoning & Backdoors)، وأخيراً نطبّق
> الدفاعات والتدريب العدائي (Adversarial Training) ونقيس الفرق قبل/بعد. كل نوت بوك يعمل على
> Google Colab المجاني، والمحتوى **بالعربية والإنجليزية**.
>
> **In English:** A practical, beginner-friendly series on **Machine Learning Security**:
> train a small CNN on MNIST, attack it with **evasion attacks (FGSM & PGD)**, explore
> **data poisoning & backdoors**, then apply **defenses (adversarial training)** and measure
> the difference before/after. Every notebook runs on free Colab and is **bilingual (AR/EN)**.

---

## 🗺️ The Learning Path (4 Projects)

| # | Project | You will learn | Runtime |
|---|---------|----------------|---------|
| [01](./01-baseline-mnist-cnn/) | **Baseline: Train a Tiny CNN** | MNIST pipeline, CNN anatomy, accuracy & model confidence — the "healthy patient" before the attack | CPU or T4 |
| [02](./02-evasion-attacks/) | **Evasion Attacks (FGSM & PGD)** | What adversarial examples are, `epsilon`, gradient-based attacks, how accuracy collapses | CPU or T4 |
| [03](./03-poisoning-backdoor/) | **Poisoning & Backdoors** | Dirty-label poisoning, trigger patterns, attack-success rate, how a backdoored model behaves | CPU or T4 |
| [04](./04-defenses-robustness/) | **Defenses & Robustness** | Adversarial training, clean-vs-robust accuracy trade-off, evaluating defenses | CPU or T4 (GPU faster) |

## 🧰 What you need

| Requirement | Details |
|---|---|
| **Basic Python & PyTorch intuition** | If you finished Project 1 of the conversational series, you are ready. No deep math needed — each notebook explains concepts with pictures and plain words. |
| **A free Google account** | For [Google Colab](https://colab.research.google.com/). CPU runtime is enough for projects 1–3; project 4 is faster on a free T4 GPU. |
| **~10 minutes per notebook** | Small CNN + MNIST = fast experiments. |

## 🧠 The attack–defense loop you will master

```text
        ┌──────────────────────── 1. BASELINE ────────────────────────┐
        │   train a small CNN on MNIST  →  clean accuracy ≈ 98%      │
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
        ┌──────────────────────── 2. EVASION ─────────────────────────┐
        │   add tiny noise (FGSM / PGD)  →  accuracy collapses        │
        │   "the image still looks like a 7, the model says 3"        │
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
        ┌──────────────────────── 3. POISONING ───────────────────────┐
        │   poison ~10% of training data with a trigger  →            │
        │   the model learns a secret backdoor                        │
        └──────────────────────────────┬──────────────────────────────┘
                                       ▼
        ┌──────────────────────── 4. DEFENSE ─────────────────────────┐
        │   adversarial training  →  robust accuracy goes up          │
        │   (clean accuracy drops a little — the trade-off)           │
        └─────────────────────────────────────────────────────────────┘
```

## 🚀 How to use this series

1. **Clone or download** the repository (or just open each notebook in Colab via the badge in every project README).
2. Run each project **in order** — project 2 attacks the model from project 1, project 3 re-trains it poisoned, project 4 defends it.
3. Every notebook is self-contained (it re-trains a tiny baseline if the model file is not found), and ships with a plain `.py` twin:
   ```bash
   pip install -q "torch" "torchvision" "matplotlib"
   python 01-baseline-mnist-cnn/baseline_mnist_cnn.py
   ```

### Repository layout

```text
AI-ML-Security/
├── 01-baseline-mnist-cnn/      # train the "patient"
├── 02-evasion-attacks/         # FGSM & PGD attacks
├── 03-poisoning-backdoor/      # data poisoning & backdoors
├── 04-defenses-robustness/     # adversarial training & evaluation
├── tools/                      # regenerate .ipynb + .py from sources
└── README.md · LICENSE · .gitignore
```

> **For maintainers:** notebooks are generated from `tools/notebook_sources/` with
> `python tools/build_notebooks.py`. Never hand-edit the `.ipynb` files.

---

## ⚠️ Educational purpose only

This series exists to **teach and defend**. Understanding how attacks work is the first
step to building secure AI systems. Use these techniques only on your own models,
public benchmark datasets (MNIST), and in research/educational settings — never on
real-world systems you do not own or without permission.

## 🤝 Contributing

Ideas welcome: add an attack (e.g., Carlini-Wagner, DeepFool), a defense (e.g., TRADES,
input preprocessing), or translate a notebook. Keep the rules: small models that run on
free Colab, bilingual (AR/EN) content, and every notebook keeps its `.py` twin.
Edit sources in `tools/notebook_sources/` and run `python tools/build_notebooks.py`.

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

**العربية:** © 2026 عبدربه العتيبي — هذا المشروع مُرخَّص بموجب رخصة MIT؛ يمكنك استخدام الأكواد وتعديلها وتعلُّمها ومشاركتها بحرية مع الإبقاء على حقوق المؤلف.

**English:** © 2026 Abdrabuh Alotaibi — this project is licensed under the MIT License; you may freely use, modify, learn from, and share the code, provided you keep this copyright notice.


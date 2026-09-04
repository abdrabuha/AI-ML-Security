# 🕳️ Project 3 — Data Poisoning & Backdoors

> **The Why:** هجمات Evasion تحدث بعد التدريب، أما هنا فالسيناريو أخطر: المهاجم يسمّم جزءاً صغيراً من بيانات التدريب (شارة + تسمية مغلوطة) فيتعلم الموديل "باباً خلفياً" سرياً — يبقى دقيقاً على البيانات النظيفة لكن أي صورة تحمل الشارة تُصنَّف حسب رغبة المهاجم.
>
> **The Why (EN):** evasion happens after training; this scenario is worse — the attacker poisons a tiny slice of the training data (trigger + wrong label) so the model learns a secret "backdoor": still accurate on clean data, yet any image carrying the trigger is classified as the attacker wishes.

<div align="center">

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abdrabuha/AI-ML-Security/blob/main/03-poisoning-backdoor/poisoning_backdoor.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)]()

`poisoning_backdoor.ipynb` &nbsp;·&nbsp; `poisoning_backdoor.py`

</div>

## 🎯 What you will learn / ماذا ستتعلم

- بناء **Trigger** (شارة 4×4) ولصقها بصور فئة المصدر / crafting & pasting a trigger
- تسميم التسميات (Dirty-label) لنحو نصف صور الرقم 7 وتحويلها إلى 9 / relabelling poisoned images
- تدريب موديل مسموم / training the poisoned model
- قياس: **Clean accuracy** (تبقى عالية!) + **Backdoor success** (ترتفع بشدة)

## 📊 Expected results / النتائج المتوقعة

| Metric | Value | Danger |
|---|---|---|
| Clean accuracy | ~95–97% | الموديل "يبدو سليماً" في الاختبار العادي |
| Backdoor success | ~90%+ | أي صورة عليها الشارة → `9` (حتى لو كانت `3`) |

## 🔬 Visual experiments / تجارب بصرية

- صورة مسمومة واضحة الشارة وتسميتها 9.
- شبكة "قبل/بعد الشارة": نفس الرقم يتنبأ به 9 فقط عند إضافة الشارة.

## 🚀 Quickstart

```bash
pip install -q torch torchvision matplotlib
python poisoning_backdoor.py
```

## 🛡️ Security takeaway / الخلاصة الأمنية

اختبار الجودة العادي **لا يكشف** الباب الخلفي أبداً — لهذا تأمين **سلسلة توريد البيانات**
(من أين تأتي بياناتك؟ من يستطيع تعديلها؟) جزء أساسي من أي مشروع ML جاد.

## ➡️ Next

**[Project 4 — Defenses & Robustness](../04-defenses-robustness/)**: نرد الضربة — التدريب العدائي يجعل الموديل يقاوم FGSM و PGD. 🛡️

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

## O'zbek tili uchun Named Entity Recognition (NER) modeli

### Model haqida
Ushbu model O'zbek tilidagi matnlarda Named Entity Recognition (NER) ni aniqlash uchun yaratilgan. Model turli xil kategoriyalardagi nomlangan entitetlarni aniqlashga qodir, jumladan shaxslar, joylar, tashkilotlar, sanalar va boshqalar. Ushbu model XLM-RoBERTa large arxitekturasiga asoslangan.

### Modelni sinab ko'rish
Model Huggingface platformasiga joylangan:
[NER Model](https://huggingface.co/risqaliyevds/xlm-roberta-large-ner)

### Diqqat!!!
Model NEWS datasetda train qilingan va asosan NEWS textlardagi NER ni aniqlay olish aniqligi baland.

### Kategoriyalar
Model quyidagi NER kategoriyalarini aniqlashga qodir:
- **LOC (Joy nomlari)**
- **ORG (Tashkilot nomlari)**
- **PERSON (Shaxs nomlari)**
- **DATE (Sana ifodalari)**
- **MONEY (Pul miqdorlari)**
- **PERCENT (Foiz qiymatlari)**
- **QUANTITY (Miqdorlar)**
- **TIME (Vaqt ifodalari)**
- **PRODUCT (Mahsulot nomlari)**
- **EVENT (Voqea nomlari)**
- **WORK_OF_ART (San'at asarlari nomlari)**
- **LANGUAGE (Til nomlari)**
- **CARDINAL (Kardinal raqamlar)**
- **ORDINAL (Ordinall raqamlar)**
- **NORP (Millatlar yoki diniy/siyosiy guruhlar)**
- **FACILITY (Inshoot nomlari)**
- **LAW (Qonunlar yoki me'yorlar)**
- **GPE (Davlatlar, shaharlar, shtatlar)**

### Misollar
Model qanday ishlashini ko'rsatish uchun bir necha misollar:
```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_name_or_path = "sizning_model_yolingiz"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path).to("cuda")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)

text = "Shavkat Mirziyoyev Rossiyada rasmiy safarda bo'ldi."
ner = nlp(text)

for entity in ner:
    print(entity)
```
Misol matni: "Shavkat Mirziyoyev Rossiyada rasmiy safarda bo'ldi."

Natijalar:
```python
[{'entity': 'B-PERSON', 'score': 0.88995147, 'index': 1, 'word': '▁Shavkat', 'start': 0, 'end': 7},
 {'entity': 'I-PERSON', 'score': 0.980681, 'index': 2, 'word': '▁Mirziyoyev', 'start': 8, 'end': 18},
 {'entity': 'B-GPE', 'score': 0.8208886, 'index': 3, 'word': '▁Rossiya', 'start': 19, 'end': 26}]
```

### Modelni yuklash va ishlatish
Modelni Hugging Face platformasidan yuklab olish va ishlatish uchun quyidagi koddan foydalanishingiz mumkin:
```python
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

model_name_or_path = "sizning_model_yolingiz"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForTokenClassification.from_pretrained(model_name_or_path).to("cuda")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
```

### Bog'lanish
Agar sizda savollar bo'lsa yoki qo'shimcha ma'lumot kerak bo'lsa, iltimos biz bilan bog'laning:
- LinkedIn: [https://www.linkedin.com/in/risqaliyevds/](https://www.linkedin.com/in/risqaliyevds/)

### Litsenziya
Ushbu model ochiq manba sifatida taqdim etiladi va barcha foydalanuvchilar uchun bepul foydalanish imkoniyatiga ega.

### Xulosa
O'zbek tili uchun NER modeli matnlarda turli xil nomlangan entitetlarni aniqlashda samarali yordam beradi. Modelning yuqori aniqligi va keng qamrovli kategoriyalari uni ilmiy tadqiqotlar, hujjatlarni tahlil qilish va boshqa ko'plab sohalarda qo'llash imkonini beradi.

# Cat and Dog Image Classifier 🐱🐶

freeCodeCamp — *Machine Learning with Python* certification project.

A convolutional neural network (TensorFlow 2 / Keras) that classifies images of
cats and dogs. It is trained on 2,000 images, validated on 1,000 images and
scored on 50 unlabelled test images. The challenge is passed with **≥ 63 %**
of the test images classified correctly (extra credit at **≥ 70 %**).

| | |
|---|---|
| Notebook | [`fcc_cat_dog.ipynb`](fcc_cat_dog.ipynb) |
| Data set | freeCodeCamp `cats_and_dogs` (2000 train / 1000 validation / 50 test) |
| Framework | TensorFlow 2 + Keras |
| Task | binary image classification (`0 = cat`, `1 = dog`) |
| Result | best validation accuracy **70.5 %**, **74 %** on a 50-image hold-out test set |

---

## কীভাবে চালাবেন (How to run)

1. **Colab খুলুন** — নিচের লিংকে ক্লিক করুন (নোটবুকটি আপনার Google Drive-এ
   `fcc_cat_dog.ipynb` নামে আপলোড করা আছে এবং লিংক-শেয়ারিং চালু আছে):
   👉 <https://colab.research.google.com/drive/16BJs03rRJzXsXZJk5UR1WC9XdEYtGg0v>
2. `Runtime → Run all` চাপুন। (GPU লাগবে না, তবে `Runtime → Change runtime type → T4 GPU`
   দিলে মাত্র ২–৩ মিনিটে ট্রেইনিং শেষ হবে।)
3. দ্বিতীয় সেলটি ডেটাসেট (`cats_and_dogs.zip`, ~230 MB) নিজে থেকেই ডাউনলোড ও
   আনজিপ করবে — প্রথমবার ১–২ মিনিট সময় নিতে পারে।
4. সব সেল চলা শেষ হলে **সবশেষ সেল**ে এমন কিছু দেখবেন:

   ```
   Your model correctly identified 70.00% of the images of cats and dogs.
   You passed the challenge!
   ```

   (আপনার রানে সংখ্যাটি কিছুটা কম-বেশি হতে পারে; ৬৩ % বা তার বেশি হলেই পাস।)
5. `File → Save a copy in Drive` দিয়ে নিজের Drive-এ কপি রাখুন, তারপর
   `Share → Anyone with the link → Viewer` করে **Colab লিংকটি** freeCodeCamp-এর
   "Solution Link" বক্সে সাবমিট করুন।

## Local / offline

```bash
pip install tensorflow matplotlib numpy
jupyter notebook fcc_cat_dog.ipynb
```

The only Colab-specific line is `%tensorflow_version 2.x` (it is inside a
`try/except`, so the notebook also runs locally and on Kaggle).

---

## ফলাফল (Result)

| মেট্রিক | মান |
|---|---|
| Training accuracy (শেষ epoch) | 64 % |
| Validation accuracy (সর্বোচ্চ) | **70.5 %** |
| ৫০টি অলেবেলড টেস্ট ইমেজের উপর | **74 % (37/50)** ✅ (পাসের শর্ত ৬৩ %) |
| ট্রেইনিং সময় | ২-কোর CPU-তে ~১০ মিনিট (Colab GPU-তে ২–৩ মিনিট) |

> সংখ্যাগুলো ৩০ epoch, batch 64, Adam (lr = 1e-4) সেটিংয়ে পাওয়া। ভ্যালিডেশন
> accuracy ৭০%-এর আশপাশে দোলাচল করে, আর মাত্র ৫০টি টেস্ট ছবিতে accuracy-র
> ±৭% পরিসংখ্যানগত অনিশ্চয়তা আছে — তাই পাসের নিশ্চয়তা বাড়াতে epoch বাড়ানো,
> BatchNormalization যোগ করা বা transfer learning (MobileNetV2) ব্যবহার করা যায়।
> তবে ৬৩%-এর শর্তে উপরের সেটিংই নিরাপদে পাস করে।

---

## মডেলের গঠন (Model architecture)

```
Input 150×150×3
 ├─ Conv2D(32, 3×3, relu) → MaxPool(2×2)
 ├─ Conv2D(64, 3×3, relu) → MaxPool(2×2)
 ├─ Conv2D(64, 3×3, relu) → MaxPool(2×2)
 ├─ Conv2D(64, 3×3, relu) → MaxPool(2×2)
 ├─ Flatten
 ├─ Dense(512, relu) → Dropout(0.5)
 └─ Dense(1, sigmoid)        # P(dog)
```

* **Optimizer:** Adam with a small learning rate (`1e-4`).
* **Loss:** binary cross-entropy.
* **Metric:** accuracy.
* **Batch size:** 64, **epochs:** 30, **steps per epoch:** 31.

### Over-fitting-এর বিরুদ্ধে দুইটি অস্ত্র

1. **Data augmentation** — `ImageDataGenerator` প্রতি epoch-এ ছবিগুলোকে এলোমেলোভাবে
   ঘোরায় (40°), shift, shear, zoom ও horizontal flip করে, তাই নেটওয়ার্ক একই ছবি
   মুখস্থ করতে পারে না।
2. **Dropout(0.5)** — ট্রেইনিংয়ের সময় অর্ধেক নিউরন বন্ধ থাকে।

### গুরুত্বপূর্ণ দুটি ট্রিক (Cell 3)

* `test/` ফোল্ডারে কোনো সাব-ফোল্ডার নেই, তাই `flow_from_directory`-এ
  `classes=['.']` দেওয়া হয়েছে — এতে ফোল্ডারটিকেই একমাত্র ক্লাস ধরা হয়
  (`Found 50 images belonging to 1 class`).
* `shuffle=False` — এতে ৫০টি টেস্ট ছবি ঠিক সেই ক্রমেই থাকে যা গ্রেডার প্রত্যাশা করে।
  এই দুটির যেকোনো একটি ভুল হলে accuracy এলোমেলোভাবে ৪০–৫০%-এ নেমে আসবে,
  যদিও ভ্যালিডেশন accuracy ৭০%+ দেখাবে।

---

## ফাইল স্ট্রাকচার

```
cats_and_dogs            # স্ক্রিপ্ট নিজেই ডাউনলোড করে
├── train
│   ├── cats  (1000)
│   └── dogs  (1000)
├── validation
│   ├── cats  (500)
│   └── dogs  (500)
└── test      (50, লেবেল নেই)
```

## রেফারেন্স

* Challenge: <https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/cat-and-dog-image-classifier>
* Boilerplate notebook: <https://github.com/freeCodeCamp/boilerplate-cat-and-dog-image-classifier>

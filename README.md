

# Machine Vision & IoT System for Palm Fruit Ripeness Classification 🍊🌴

A deep-learning pipeline (TensorFlow + Xception) that classifies oil-palm Fresh Fruit Bunches (FFB) into **unripe, underripe, and ripe** classes with ≈96% top-1 accuracy.
The model powers a Gradio webcam GUI for real-time predictions and can be embedded in a low-cost LED-lit inspection box for field deployment.

---

## ✨ Key Features

| Feature              | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| Transfer-learning core | Pre-trained **Xception** backbone fine-tuned on 128 × 128 RGB crops    |
| Lightweight dataset    | 670 images after augmentation – 537 train / 133 val over 3 classes      |
| High accuracy          | 96% train / 83% val after 150 epochs, F1≈0.86                           |
| Explainability         | Integrated Grad-CAM visualisation for model interpretability            |
| Zero-code GUI          | One-click Gradio interface (`python gui/app.py`)                        |
| Portable prototype     | 3 × AA-powered LED chamber + smartphone camera for consistent images    |

---

## 🗄️ Repository Structure

```
├── notebooks/                # Colab / Jupyter training notebooks
├── data/
│   ├── ripe/                # place images here
│   ├── underripe/
│   └── unripe/
├── src/
│   ├── train.py             # end-to-end training script
│   ├── predict.py           # CLI prediction utility
│   └── utils/               # data loaders, metrics, vis
├── gui/
│   └── app.py               # Gradio webcam demo
├── hardware/                # CAD, wiring diagram, BOM
├── models/                  # saved `.h5` weights
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/<your-handle>/palm-fruit-ripeness.git
cd palm-fruit-ripeness

# 2. Create env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # TensorFlow 2.x, Gradio, OpenCV …

# 3. Prepare dataset
#   data/
#     └── <class_name>/image01.jpg ...
python src/utils/split_train_val.py --root data --val_split 0.2

# 4. Train
python src/train.py --epochs 150 --img_size 128 --batch 16

# 5. Run the webcam GUI
python gui/app.py   # opens http://127.0.0.1:7860
```

---

## 📦 Dataset

- **Sources:** Self-captured smartphone images + public Kaggle FFB photos
- **Classes:**
  - `0 = unripe` (black/purple)
  - `1 = underripe` (reddish)
  - `2 = ripe` (orange)
- **Augmentations:** ±45° rotation & 0.5–1.0 zoom to combat overfitting

---

## 🏗️ Model Architecture

```
Input(128×128×3)
│
└── Xception(base_weights="imagenet", include_top=False) → 4×4×2048
     └── GlobalAveragePooling
          └── Dense(512) + ReLU
               └── Dense(3) + Softmax
```

See full plot in the report (Fig. 4.3) for layer-level detail.

---

## 🖥️ Hardware Prototype (optional)

| Part                 | Qty                 | Notes                                     |
|----------------------|---------------------|-------------------------------------------|
| Foam-board light box | 1                   | 300 mm cube, matte-white interior         |
| RGB LED strip        | 1 m                 | 500 lux inside chamber                    |
| Smartphone camera    | 1                   | tested on Galaxy S9 / Iriun Webcam driver |
| Power                | 3 × AA rechargeable | ≈6 h runtime                              |

Assembly guides are in `hardware/`.

---

## 🔬 Results

| Metric   | Train      | Validation |
|----------|------------|------------|
| Accuracy | **96.8%**  | **83.5%**  |
| F1-score | 0.97       | 0.86       |

Confusion matrices and Grad-CAM maps are provided in the `/docs` folder for reproducibility.

---

## ✍️ How to Cite

If you use this code or data, please cite —

```
@report{AlKuhali2023PalmRipeness,
  title  = {Machine Vision and IoT System for Palm Fruit Ripeness Classification},
  author = {Mohammed Hashem Mohammed Al-Kuhali},
  school = {UCSI University},
  year   = {2023}
}
```

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

## 🛡️ License

MIT – see `LICENSE` for details.

---

# 🧠 Brain Tumor Segmentation on BRISC2025 (U-Net)

A deep learning project that performs **pixel-wise brain tumor segmentation** on MRI scans using a U-Net (ResNet34 encoder, ImageNet-pretrained), trained on the **BRISC2025** dataset.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

While classification tells us *what type* of tumor is present, segmentation tells us *exactly where* it is — pixel by pixel. This project fine-tunes a **U-Net** with a pretrained ResNet34 encoder to produce binary tumor masks from T1-weighted brain MRI slices, evaluated with Dice score and IoU.

## 📊 Dataset — BRISC2025

- **Source:** [BRISC2025 on Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025/) (Fateh et al., 2026, *Scientific Data*)
- **Samples:** 6,000 T1-weighted MRI slices (5,000 train / 1,000 test)
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Planes:** Axial, Coronal, Sagittal
- **Masks:** Physician-reviewed, pixel-wise binary segmentation masks

```
brisc2025/
└── segmentation_task/
    ├── train/
    │   ├── images/   (*.jpg)
    │   └── masks/    (*.png, binary)
    └── test/
        ├── images/
        └── masks/
```

Image and mask filenames share the same basename (e.g. `brisc2025_train_00001_gl_ax_t1.jpg` ↔ `.png`).

## 🧠 Methodology

1. **Data Loading** — paired image/mask loading via a custom `Dataset`, matched by filename
2. **Augmentation** — resize to 256×256, horizontal flip, rotation, brightness/contrast jitter (Albumentations)
3. **Model** — U-Net with a ResNet34 encoder (ImageNet-pretrained), 1-channel sigmoid output
4. **Loss** — combined **Dice + BCE loss**, standard for imbalanced medical segmentation masks
5. **Training** — Adam optimizer with `ReduceLROnPlateau` scheduling, best model checkpointed on validation Dice
6. **Evaluation** — Dice coefficient, IoU (Jaccard index), qualitative image/mask/prediction comparisons

## 📈 Results

<img width="1189" height="440" alt="image" src="https://github.com/user-attachments/assets/1c6f56ae-2759-40b8-93f1-16c7e80ced8c" />


<img width="866" height="1190" alt="image" src="https://github.com/user-attachments/assets/4b65a373-73b5-49ac-868c-ba964e6cf7f4" />


> ⚠️ **Placeholder plots.** The two images above were generated from a quick 3-epoch
> smoke test on tiny synthetic (random-noise) images — used only to verify the
> pipeline runs end-to-end without errors. They do **not** reflect real BRISC2025
> performance. Run `src/train.py` (or the notebook) on the actual dataset and
> replace these files with your real results before publishing.

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Framework:** PyTorch, `segmentation-models-pytorch` (U-Net + pretrained encoders)
- **Augmentation:** Albumentations
- **Libraries:** numpy, matplotlib, Pillow
- **Environment:** Kaggle Notebook (GPU recommended)

## 📂 Project Structure

```
BRISC2025-Tumor-Segmentation/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   └── brisc2025_unet_segmentation.ipynb
│
├── src/
│   ├── utils.py
│   ├── train.py
│   └── predict.py
│
├── models/              ← created by train.py (unet_brisc_seg.pth)
│
└── results/
    ├── loss_dice_curves.png
    └── sample_predictions.png
```

## ⚙️ Installation & Usage

```bash
git clone https://github.com/sagor5271/BRISC2025-Tumor-Segmentation.git
cd BRISC2025-Tumor-Segmentation
pip install -r requirements.txt
```

**Run on Kaggle (recommended):**
Upload `notebooks/brisc2025_unet_segmentation.ipynb` to Kaggle, attach the BRISC2025 dataset, and run all cells. The notebook expects the dataset at:
```
/kaggle/input/datasets/briscdataset/brisc2025/brisc2025
```

**Train locally / on a GPU machine:**
```bash
python src/train.py --data_root /path/to/brisc2025 --epochs 20 --batch_size 16
```

**Predict a mask for a new MRI image:**
```bash
python src/predict.py --image path/to/scan.jpg --output mask_out.png
```

## 🚀 Future Work

- Multi-class segmentation (separate masks per tumor type) instead of binary tumor/background
- Try U-Net++ / DeepLabV3+ / Swin-UNETR for comparison
- Combine with the classification model (`glioma`/`meningioma`/`pituitary`/`no_tumor`) for a joint classify-and-localize pipeline
- 5-fold cross-validation for more robust Dice/IoU estimates

## 👤 Author

**Md Sagor Hossain**
Biomedical Engineering, Islamic University, Bangladesh
📧 sagor.bme.iu@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/md-sagor-hossain-8471ab336)

## 📄 License

MIT License — see [LICENSE](LICENSE).

## ⚠️ Disclaimer

Developed for research and educational purposes only. Not intended for clinical diagnostic use.

## 📄 Dataset Citation

```
@article{Fateh_2026,
  title={BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification},
  volume={13}, ISSN={2052-4463},
  DOI={10.1038/s41597-026-06753-y},
  number={1}, journal={Scientific Data},
  publisher={Springer Science and Business Media LLC},
  author={Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and Rezvani, Sadjad and Fateh, Fatemeh and Fateh, Mansoor and Abolghasemi, Vahid},
  year={2026}, month=Feb
}
```

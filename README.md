# ABAW Emotion Recognition Model

This repository contains the official implementation of the following paper:

> S. Min, J. Yang and S. Lim, "Emotion Recognition Using Transformers with Random Masking," _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)_, Seattle, WA, USA, 2024, pp. 4860-4865, doi: [10.1109/CVPRW63382.2024.00489](https://ieeexplore.ieee.org/document/10678303).

This project provides training code for a temporal emotion recognition model based on ViT and Transformer, developed for the [ABAW Challenge](https://ibug.doc.ic.ac.uk/resources/abaw/) across EXPR, VA, and AU tasks.

📄 [한국어 문서 보기 (View in Korean)](Readme_ko.md)

---

## 📁 Data Directory Structure

```
data/
├── raw/
│   └── images/                  # Cropped-aligned face images provided by the ABAW organizers
│       └── <video_id>/          # e.g., 1-30-1280x720
│           └── 00001.jpg
│           └── ...
│
├── features/
│   └── features_mean/           # or features_cls/
│       └── <video_id>/
│           └── 00001.npy        # Feature from ViT CLS/Mean token (numpy, [1, 768])
│           └── ...
│
└── labels/
    ├── EXPR_Recognition_Challenge/
    │   └── Train_Set/
    │       └── 1-30-1280x720.txt    # Integer labels (emotion classes)
    ├── VA_Estimation_Challenge/
    │   └── Train_Set/              # valence, arousal ∈ [-1, 1], comma-separated
    └── AU_Detection_Challenge/
        └── Train_Set/              # 12 AU labels per frame, values are 0/1 or -1
```

### 📌 Notes
- **Raw images**: cropped and aligned face images provided by the challenge organizers.
- **Features**: token embeddings (CLS or Mean) extracted from ViT models.
- **Labels**: formats vary by task; the first line of each file is a header.
- Missing frames are automatically handled during dataset construction.
- Invalid labels (`-1`, `-5`) are ignored during training.

---

## 🚀 How to Train

```bash
python train.py --config config/config.yaml
```

For model architecture, loss function, and training configuration, refer to `config/config.yaml`.
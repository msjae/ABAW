# ABAW 감정 인식 모델

본 Repository는 다음 논문의 코드입니다:

> S. Min, J. Yang and S. Lim, "Emotion Recognition Using Transformers with Random Masking," _2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)_, Seattle, WA, USA, 2024, pp. 4860-4865, doi: [10.1109/CVPRW63382.2024.00489](https://ieeexplore.ieee.org/document/10678303).

이 프로젝트는 [ABAW Challenge](https://ibug.doc.ic.ac.uk/resources/abaw/)의 EXPR, VA, AU 태스크를 기반으로, ViT와 Transformer를 사용한 시계열 감정 인식 모델 학습 코드입니다.

📄 [View in English](README.md)

---

## 📁 데이터 하위 구조

```
data/
├── raw/
│   └── images/                  # 주최 측에서 제공한 얼굴 정렬 이미지 (cropped-aligned)
│       └── <video_id>/          # 예: 1-30-1280x720
│           └── 00001.jpg
│           └── ...
│
├── features/
│   └── features_mean/           # 또는 features_cls/
│       └── <video_id>/
│           └── 00001.npy        # ViT의 CLS/Mean 토큰 기반 특징 (numpy, [1, 768])
│           └── ...
│
└── labels/
    ├── EXPR_Recognition_Challenge/
    │   └── Train_Set/
    │       └── 1-30-1280x720.txt    # 정수 레이블 (감정 클래스)
    ├── VA_Estimation_Challenge/
    │   └── Train_Set/           # valence, arousal ∈ [-1, 1], 쉼표 구분
    └── AU_Detection_Challenge/
        └── Train_Set/           # 12개의 AU 레이블, 값은 0/1 또는 -1
```

### 📌 참고 사항
- **Raw 이미지**: 주최 측에서 제공한 얼굴 정렬 이미지 사용
- **Features**: ViT 모델로 추출한 CLS 또는 Mean 토큰 특징 벡터
- **Label**: task에 따라 형식이 다르며, 첫 줄은 항상 헤더
- 일부 프레임 누락 가능 → `Dataset`에서 자동 정렬 및 필터링
- 무효 레이블 (`-1`, `-5`)은 학습 시 자동 무시

---

## 🚀 학습 실행

```bash
python train.py --config config/config.yaml
```

모델 구조, 손실 함수, config 구성은 `config/config.yaml`을 참고하세요.

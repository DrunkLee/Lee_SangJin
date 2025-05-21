# 이상진 - 포트폴리오 소스 코드

📌 **이 저장소에 포함된 모든 코드는 본인이 직접 작성한 코드로만 구성되어 있으며, 타인의 소유권이 있는 파일이나 외부 프레임워크 코드는 포함되어 있지 않습니다.**

[🔗 포트폴리오 PDF 링크](https://github.com/DrunkLee/Lee_SangJin/blob/main/%EC%9D%B4%EC%83%81%EC%A7%84_%ED%8F%AC%ED%8A%B8%ED%8F%B4%EB%A6%AC%EC%98%A4_V1.1.pdf)

---

## 📁 프로젝트 목록

### 1. Video to Text / Text to Frame (V2T_T2V)

> 자연어로 장면을 검색하거나, 장면에 대한 설명을 생성하는 멀티모달 시스템입니다.

🔗 **GitHub**: [boostcamp_AI_Tech_hackathon_TVING](https://github.com/DrunkLee/boostcamp_AI_Tech_hackathon_TVING)

- ✅ 주요 기여
  - InternVL 2.5 기반 LoRA Fine-Tuning 적용 *(코드 미포함)*
  - FFmpeg 및 OpenCV 기반 장면 분할 및 추론 모듈 구현 (`models/analyze.py`)
  - Wav2Vec 기반 STT 모듈 구현 (`models/audio_model.py`)
  - VLM 추론 결과 번역 기능 (영↔한) 개발 (`models/translation.py`)
  - Streamlit 기반 사용자 인터페이스 구현 (`main.py`, `modules/`)
  - Pseudo Labeling 실험 및 적용 *(코드 미포함)*

---

### 2. Hand Bone Segmentation (HandBoneSeg)

> X-ray 손뼈 이미지에 대한 Segmentation 모델을 개발한 프로젝트입니다.

🔗 **GitHub**: [boostcamp_AI_Tech_semanticsegmentation](https://github.com/DrunkLee/boostcamp_AI_Tech_semanticsegmentation)

- ✅ 주요 기여
  - 다양한 Loss 조합 실험 및 구현 (`loss/`)
  - TensorFlow 기반 모델을 PyTorch로 변환하여 구현 (`models/DUCKNet/`)
  - [Spatial & Channel Attention](https://arxiv.org/pdf/1807.06521)을 적용한 Custom UNet 설계 (`models/CUSTOM/`)

---

### 3. 다국어 영수증 OCR

> 다국어로 작성된 영수증 내 텍스트 영역을 감지하는 OCR Detection 프로젝트입니다.  
> 본 과제는 **Baseline 수정이 금지된 Data-Centric AI 프로젝트**입니다.

🔗 **GitHub**: [boostcamp_AI_Tech-datacentric](https://github.com/DrunkLee/boostcamp_AI_Tech-datacentric)

- ✅ 주요 기여
  - Annotation 가이드라인 수립 및 전체 데이터 정제
  - REST API를 활용한 학습 현황 실시간 알림 기능 구현 (`services/`)

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

---

### 4. AI Competition
> 다양한 산업 분야의 실제 데이터를 다루며 문제 해결 능력을 기르기 위해 AI 경진대회에 꾸준히 참여하고 있습니다. 데이터 정제부터 모델링, 성능 개선까지 End-to-End로 프로젝트를 수행하며 실전 역량을 강화하고 있습니다.

#### 1) 식음업장 메뉴 수요 예측 (Private 55위)
> 시계열 특성을 가진 식음업장 데이터를 분석하여 메뉴별 수요를 예측하는 회귀 문제입니다.

🔗 **대회 링크**: [식음업장 메뉴 수요 예측](https://dacon.io/competitions/official/236559/overview/description)
- 결과
    - Public  : [**39** / 820]
    - Private : [**55** / 820]

#### 2) 고객 지원 등급 분류 (Private ??)
> 고객의 서비스 이용 데이터와 계약 정보를 분석하여, 지원 필요 수준을 사전에 예측하는 분류 문제입니다.

🔗 **대회 링크**: [고객 지원 등급 분류](https://dacon.io/competitions/official/236562/overview/description)
- 결과
    - Public  : [**??** / ??]
    - Private : [**??** / ??]

#### 3) K intelligence 해커톤 2025 <프롬프트엔지니어링> (Private ??)
> 참가자는 K intelligence의 GPT4o 기반 Custom 모델(beta)을 활용해, 유저 프롬프트만으로 문제를 해결할 수 있는 시스템 프롬프트를 설계해야 합니다.

🔗 **대회 링크**: [K intelligence 해커톤 2025](https://dacon.io/competitions/official/236552/overview/description)
- 결과
    - Public  : [**??** / ??]
    - Private : [**??** / ??]
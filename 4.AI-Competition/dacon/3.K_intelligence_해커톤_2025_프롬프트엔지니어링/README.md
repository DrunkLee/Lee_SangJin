# [K intelligence 해커톤 2025 <프롬프트 엔지니어링>](https://dacon.io/competitions/official/236552/overview/description)

## 배경
참가자는 K intelligence의 GPT4o 기반 Custom 모델(beta)을 활용해, 유저 프롬프트만으로 문제를 해결할 수 있는 시스템 프롬프트를 설계해야 하며, 프롬프트의 기획력, 문제 해결 전략, 응답의 일관성과 품질 등이 종합적으로 평가됩니다.

## 데이터
```
samples.csv
```

1. `sample.csv`
- user_prompt : 입력 문장
- output : 입력 문장에 대해 4가지 속성으로 분류한 출력 결과

## 평가

<img width="466" height="328" alt="Image" src="https://github.com/user-attachments/assets/cdbb9d61-14ac-426d-8337-f38b02d14804" />

> - Public score : 전체 테스트 데이터 100%
> - Private score : 예선 종료 시점의 Public Score

- **평가 산식**(1점 만점)
  - 모델 분류 정확도 (80%)
  - 시스템 프롬프트 한글 비율 점수 (10%)
  - 시스템 프롬프트 길이 점수 (10%)

- **모델 분류 정확도**
  - 유형, 극성, 시제, 확실성 4개 속성에 대한 각각의 정확도 평균

- **시스템 프롬프트 한글 비율 점수**
  - 시스템 프롬프트의 전체 문자 수(공백·줄바꿈 제외) 대비 한글 문자 비율 점수

- **시스템 프롬프트 길이 점수**

## 일정
- 2025년 08월 25일(월) 10:00 ~ 2025년 09월 10일(수) 10:00

## 결과
- Public  : [?? / ??]
- Private : [?? / ??]

## 파일
```
├── main.py
├── preprocess.py
├── prompt_analysis.py
├── requirements.txt
└── analysis_report_YYYYMMDD_HHMMSS.txt

```
`main.py`: 전체 데이터 처리 및 분석 파이프라인을 실행하는 메인 스크립트입니다.

`preprocess.py`: DataPreProcess 클래스가 정의된 모듈입니다. CSV 파일을 불러와 전처리하고, 중간 결과물(*.pickle)을 저장/로드하는 기능을 담당합니다.

`prompt_analysis.py`: PromptAnalysis 클래스가 정의된 모듈입니다. 전처리된 데이터를 바탕으로 형태소 분석 및 카테고리별 통계 리포트를 생성하는 핵심 로직을 포함합니다.

`analysis_report_*.txt`: main.py 실행 시 최종적으로 생성되는 분석 결과 리포트입니다. 파일명에 실행 시점의 타임스탬프가 포함됩니다.
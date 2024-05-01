# Bul4jo 알파코2차\_미니프로젝트\_4조

---

### 최종 목표

---

### Whisper base 모델을 AI Hub 소음 데이터에 대해 fine-tuning 진행하고, 진행 전/후 모델의 성능을 비교 분석

---

### 진행 완료

---
- ### Local 개발 환경 설정
  - 학습 데이터 용량(약1.0TB)으로 인해 로컬에서 개발 진행
  - 아나콘다 가상환경을 활용해 개별 프로젝트에 적합한 개발 환경 세팅
  - 많은 데이터를 효율적으로 학습하기 위해 GPU 연결 및 활용
  - Tensorflow 활용 및 VSCode Jupyter Notebook 활용
- ### AI Hub 소음 데이터 다운로드 (Noise data)
- ### AI Hub 소음 데이터 정제
  - AI Hub Noise data 중 wav파일은 NV파일(Noise O)과 VN파일(Noise X)이 혼재되어 있어 이 중 VN파일은 모두 제외함.
  - AI Hub Noise data 중 text파일은 json파일과 srt파일로 이루어져 있고, 이 중 srt파일에서 pysubs2 라이브러리를 활용해 텍스트 추출함.
- ### Whisper base 모델 파인튜닝 진행 전 성능 측정
  - Ver 0.
    - Base 모델을 활용한 적절한 inference 방법을 찾지 못한 상태라, Whisper fine-tuning^[1]^을 참조하여 테스트셋으로 파인튜닝을 진행하되
      train/test 비율을 1:999로 설정하고 trainer.evaluate() 결과값을 확인하여 실제로 학습은 거의 진행하지 않은 상태에서 결과값을 보고자 함.
      ![image](https://github.com/Ijjoe/Bul4jo/assets/161268753/14fa218d-7085-4360-8e93-63ea772249a4)

    - ^[1]^ https://velog.io/@mino0121/NLP-OpenAI-Whisper-Fine-tuning-for-Korean-ASR-with-HuggingFace-Transformers
  - Ver 1.
    - Whisper-base 모델 및 pipeline 모듈을 활용하여 테스트셋에 대해 inference 성능 측정
- ### Noise data 학습
  - 학습 데이터(약 20,000건)가 충분하다고 판단되어 train/valid 비율은 9:1로 설정함.
  - ![image](https://github.com/Ijjoe/Bul4jo/assets/161268753/1c7a70a5-78f1-4e3d-ad94-0672f1b3490a)
- ### Noise model 평가
  - 원본 wav 파일의 길이가 4~5분인데 비해 모델 입력은 이 중 30초만을 이용함.
  - 음성 데이터 입력은 30초 분량이고 라벨은 원본 전체 길이에 대응하기 때문에 input-output이 적절히 매칭되지 않아 모델의 평가가 제대로 이루어지지 않음. 
  - 이를 해결하기 위해선 전체 음성을 모델 input으로 사용하거나 input 길이(30초)에 맞게 라벨 데이터를 전처리 할 필요가 있음.

---

### 현재 진행 중 (~24/05/02)

---
- ### AI Hub 방언 데이터 다운로드 (Accent data)

---

### 진행 예정

---
- ### 소음 데이터 라벨 변경 후 재학습 및 평가
  - json 파일 메타데이터 중 startTime 및 endTime을 활용하여 0~30초 구간의 전사 text를 추출한 후 이를 라벨로 재학습 및 평가 진행
- ### AI Hub 방언 데이터 학습 및 평가
  - 방언 데이터는 길이가 짧은 wav 파일로 구성되어 있어 소음데이터와 같은 문제가 발생하지 않을 것으로 예상














# 테스트!
---

#### 프로젝트를 소통을 위한 마크다운 사용법 (복사해서 쓰세요)

---

```

(copy 가능)
# 가장큰 크기의 text로 변환

## 그다음 작은 크기위 text로 변환

### 그다음 작은 크기의 text로 변환

#### 그다음 작은 크기의 text로 변환

##### 그다음 작은 크기의 text로 변환

###### 그다음 작은 크기의 text로 변환
```

&downarrow;&downarrow;&downarrow;&downarrow;(아래는 적용한 모습)

# 가장큰 크기의 text로 변환

## 그다음 작은 크기위 text로 변환

### 그다음 작은 크기의 text로 변환

#### 그다음 작은 크기의 text로 변환

##### 그다음 작은 크기의 text로 변환

###### 그다음 작은 크기의 text로 변환

---

```
(copy 가능)
code 1

     code 2 // code block 할 내용

code 3 // 정상적으로 출력
```

---

&downarrow;&downarrow;&downarrow;&downarrow;(아래는 적용한 모습)
code 1

     code 2 // code block 할 내용

code 3 // 정상적으로 출력

---

```
(copy 가능)
BlockQuote 사용하기
> 안녕하세요
>
> > 저는
> >
> > > bul 입니다.
```

---

&downarrow;&downarrow;&downarrow;&downarrow;(아래는 적용한 모습)

> 안녕하세요
>
> > 저는
> >
> > > bul 입니다.

---

```
(copy 가능)
1. 안녕하세요
2. 오늘하루도
3. 행복하세요
```

---

&downarrow;&downarrow;&downarrow;&downarrow;(아래는 적용한 모습)

1. 안녕하세요
2. 오늘하루도
3. 행복하세요

---

```
(copy 가능)
- 안녕
  - Hello
    - hi

* 안녕
  - Hello
    - hi

- 안녕
  - Hello
    - hi
```

&downarrow;&downarrow;&downarrow;&downarrow;(아래는 적용한 모습)

- 안녕
  - Hello
    - hi

* 안녕
  - Hello
    - hi

- 안녕
  - Hello
    - hi

```

```

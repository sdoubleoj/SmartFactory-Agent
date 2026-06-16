# AI4I 2020 예지보전 데이터 스키마

## 데이터셋 개요

| 항목 | 내용 |
|---|---|
| 출처 | UCI Machine Learning Repository (Dataset #601) |
| 원저자 | Stephan Matzka (HTW Berlin, 2020) |
| 대상 | 산업용 밀링 머신 공정 |
| 샘플 수 | 10,000행 |
| 컬럼 수 | 14개 |
| 형식 | 헤더 포함 CSV |
| 결측값 | 없음 |

---

## 파일 구성

| 파일명 | 역할 |
|---|---|
| `ai4i2020.csv` | 전체 데이터 단일 파일 (train/test 미분리) |

> train/test 분리는 수행팀이 직접 구성 (일반적으로 80:20 랜덤 분할 또는 시계열 순서 분할)

---

## 컬럼 구성 (총 14개)

### 식별자 (2개)

| # | 컬럼명 | 타입 | 설명 |
|---|---|---|---|
| 1 | `UDI` | int | 고유 행 번호 (1~10,000) |
| 2 | `Product ID` | str | 제품 고유 ID. 앞 글자가 Type과 동일 |

### 범주형 입력 (1개)

| # | 컬럼명 | 타입 | 값 | 설명 |
|---|---|---|---|---|
| 3 | `Type` | str | L / M / H | 제품 품질 등급. L(저) 50% / M(중) 30% / H(고) 20% |

### 수치형 입력 — 공정 변수 (5개)

| # | 컬럼명 | 타입 | 단위 | 설명 |
|---|---|---|---|---|
| 4 | `Air temperature [K]` | float | K (켈빈) | 공기 온도. 정규분포 근사, 평균 300K |
| 5 | `Process temperature [K]` | float | K (켈빈) | 공정 온도. 공기 온도 + 약 10K 오프셋 |
| 6 | `Rotational speed [rpm]` | int | rpm | 회전수. 2860rpm 기준 정규분포 |
| 7 | `Torque [Nm]` | float | Nm | 토크. 40Nm 기준 정규분포, 음수 없음 |
| 8 | `Tool wear [min]` | int | 분 | 공구 누적 사용 시간. 품질 등급별 마모 속도 상이 (H<M<L) |

### 레이블 (6개)

| # | 컬럼명 | 타입 | 값 | 설명 |
|---|---|---|---|---|
| 9 | `Machine failure` | int | 0 / 1 | **메인 타깃.** 아래 5종 고장 중 하나라도 발생 시 1 |
| 10 | `TWF` | int | 0 / 1 | Tool Wear Failure — 공구 마모 고장 |
| 11 | `HDF` | int | 0 / 1 | Heat Dissipation Failure — 방열 고장 |
| 12 | `PWF` | int | 0 / 1 | Power Failure — 전력 고장 |
| 13 | `OSF` | int | 0 / 1 | Overstrain Failure — 과부하 고장 |
| 14 | `RNF` | int | 0 / 1 | Random Failure — 무작위 고장 (원인 없음) |

---

## 클래스 분포

### Machine failure (전체)

| 클래스 | 건수 | 비율 |
|---|---|---|
| 정상 (0) | 9,661 | 96.6% |
| 고장 (1) | 339 | 3.4% |

> **심한 클래스 불균형** — SMOTE, class_weight, threshold 조정 등 대응 필요

### 고장 모드별 발생 건수

| 고장 모드 | 컬럼 | 건수 | 설명 |
|---|---|---|---|
| 방열 고장 | HDF | 115 | 가장 많음 |
| 과부하 고장 | OSF | 98 | |
| 전력 고장 | PWF | 95 | |
| 공구 마모 고장 | TWF | 46 | |
| 무작위 고장 | RNF | 19 | 가장 적음, 예측 불가 |

> 고장 모드 합계(373) > Machine failure(339) : 복수 고장 모드가 동시에 발생한 행 존재

---

## 고장 발생 조건 (데이터 생성 규칙)

데이터가 합성(synthetic)이므로 고장 발생 조건이 명시되어 있음

| 고장 모드 | 발생 조건 |
|---|---|
| TWF | 공구 마모가 200~240분 사이에서 랜덤하게 고장 |
| HDF | 온도차(공정-공기) < 8.6K 이고 회전수 < 1380rpm |
| PWF | 토크 × 회전수(각속도 환산)가 정상 범위 벗어남 |
| OSF | 공구 마모 × 토크가 등급별 임계값 초과 (H:11,000 / M:12,000 / L:13,000) |
| RNF | 0.1% 확률로 무작위 발생 |

---

## 예측 과제 구성

| 과제 | 타깃 컬럼 | 유형 |
|---|---|---|
| 기본 | `Machine failure` | 이진 분류 (정상 vs 고장) |
| 확장 | `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | 다중 레이블 분류 (고장 원인 유형 구분) |

---

## 파생 변수 아이디어

```python
# 물리적으로 의미 있는 조합
df["Power [W]"]     = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * 3.14159 / 60)
df["Temp diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]
df["Wear_Torque"]   = df["Tool wear [min]"] * df["Torque [Nm]"]  # OSF 조건과 직결
```

---

## pandas 로드 코드

```python
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("ai4i_data") / "ai4i2020.csv")

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
LABEL_COL  = "Machine failure"
FAULT_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

print(df.shape)       # (10000, 14)
print(df[LABEL_COL].value_counts())
```
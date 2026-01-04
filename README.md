# Coin Detection & Classification (OpenCV)

조명/배경 변화가 있는 이미지에서 **동전을 검출**하고, 검출된 동전을 **10/50/100/500원으로 분류**하는 프로젝트.

조건: 동전이 겹치는 경우는 없음, 동전 이외의 원형 물체는 없음, 입력 사진의 해상도가 다양함, 촬영 배율의 가변성

## Overview
- **Coin Detection**: 전처리 + 이중 마스크 + (붙은 동전) 분리/원 복원 + 중복 제거
- **Coin Classification**: 반지름 기반 Easy/Hard 모드 분기 후, 크기/색상/템플릿을 단계적으로 융합

---

## 1) Coin Detection Pipeline

### Preprocessing
1. `Grayscale → CLAHE → Gaussian Blur`
2. `Auto-Canny → Morphological Close`

### Dual Mask (조명/배경 대응)
- **Adaptive mask**: adaptive threshold + close  
- **Otsu mask**: Otsu + close 후, **Adaptive(dilate)로 가이드**해서 AND (그림자/오검출 억제)

### Circle Candidates (마스크별 동일 파이프라인)
1. **붙은 동전 분리**: Distance Transform 기반 중심 후보 + **Watershed**
2. **단일/복수 판별**: Circularity / Solidity / Peak count로 상태 결정
3. **원 추정 분기**
   - **Single**: contour 기반 `minEnclosingCircle`
   - **Touching**: `Hough Gradient` (반지름 구간 분할 + param2 강→약 탐색)  
     + 에지/마스크 일치율로 신뢰도 평가  
     + 지역 탐색으로 중심/반지름 미세 보정
4. **후처리**: 두 마스크 결과 **통합 + 중복 제거 + 큰 원 내부의 작은 원 제거**

---

## 2) Coin Classification Pipeline

### Easy / Hard 모드 분기
- 반지름 그룹(클러스터) 수가 **4개 이상**으로 명확하면 → **Easy(크기만으로 분류)**
- 그 외 → **Hard(크기 + 색상 + 템플릿 융합)**

### Easy Mode (크기 기반)
- REL_MAP(동전 상대 크기 비율) 기준으로 전체 스케일 `s`를 추정
- 각 동전 반지름 `r`을 `s × REL_MAP[class]`에 매칭해 라벨 결정

### Hard Mode (크기 + 재질/색 + 템플릿)
1. `ratio = r / s` 기반으로 반지름 비율 클러스터링
2. **구리/은색 판별(10원 우선 확정)**  
   - ROI 내부 vs 주변 링 영역의 HSV/LAB 통계 차이로 **copper score** 계산
3. **스케일 재추정**  
   - 은색(50/100/500) 동전 반지름을 우선 사용해 `s` 재계산
4. 은색 계열 최종 라벨
   - 은색 그룹이 3개면: 평균 반지름 작은 순서대로 **50 → 100 → 500**
   - 1~2개면: **ORB 템플릿 매칭 점수 + 반지름 근접도**를 함께 사용

---

## Key Parameters (default)
| Parameter | Value | Note |
|---|---:|---|
| BLUR_K | 11 | Blur 커널(잡음 완화) |
| SIGMA | 0.33 | Auto-Canny 임계값 자동 조절 |
| CLOSE_K | 3 | 끊긴 에지 연결 |
| MIN_COIN_AREA | 2000 px | 작은 잡음 제거 |
| HOUGH_DP | 1.2 | 허프 해상도 스케일 |
| HOUGH_PARAM1 | 120 | 허프 내부 Canny 상위 임계 |
| HOUGH_PARAM2 | 22 (sweep) | 강→약 단계 조절 |
| guide_dilate | 17~21 px | Otsu를 Adaptive 내부로 제한 |
| peak threshold | 0.55×max | 붙은 동전 판단 기준 |
| C10_SCORE_THR | 0.52 | 10원(구리) 판별 임계 |

---

## Limitations / Failure Cases
- 원 검출이 불안정(강한 그림자/반사/저조도)하면 반지름이 과소/과대 추정되어 분류까지 영향
- Otsu 단독 사용 시 그림자 경계가 원으로 오검출될 수 있어 **Adaptive 가이딩으로 보완**
- 붙은 동전은 붙임이 심할수록 분리 안정성 저하 가능
- ORB 템플릿 매칭은 회전/마모/조명/앞면-뒷면 차이로 성능이 흔들릴 수 있어 **보조 신호로만 사용**

---


<img width="334" height="257" alt="image" src="https://github.com/user-attachments/assets/e70074b7-7516-4b90-9de9-14b324ab3aa7" />
<img width="334" height="257" alt="image" src="https://github.com/user-attachments/assets/d1c260b7-71a6-4841-a744-7b7143993cd8" />


## Tech Stack
- Python, OpenCV (CLAHE / Canny / morphology / watershed / HoughCircles / ORB)

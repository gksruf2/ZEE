# ZEE (Zoom-Enhance-Enhance): Remote Sensing Image Super-Resolution

> **TL;DR** — AID(x4, bicubic)에서 **TTST/HAT/SRCNN**을 공정 비교하고, WorldStrat(pair)용 **전처리 파이프라인**(HM/NoHM, **SSIM Top-40%**, **LightGlue+DISK** 정합 필터, **RGB-SAD** 방사 일치)을 구축해 RS-ISR 학습 안정성과 성능 변화를 검증했습니다.

---

## 1) 프로젝트 개요
- **프로젝트명**: ZEE (Zoom-Enhance-Enhance)
- **기간**: 2025-07 ~ 2025-09
- **레포 상태**: <!-- TODO: 공개/비공개 및 라이선스(MIT/Apache-2.0 등) -->
- **팀/역할**: <!-- TODO: 이름 – 핵심 담당 -->
- **핵심 기여**
  - **팀**: AID에서 TTST/HAT/SRCNN **동일 조건 비교**, **엣지 모듈** / **MAE 전이** 실험
  - **개인**: WorldStrat **정합/방사 전처리** + **난이도 필터(SSIM Top-40%)** + **RGB-SAD** 기준 정립

---

## 2) 레포 구조
```
zee/
 ├─ README.md
 ├─ configs/
 │   ├─ aid_x4_bicubic.yml
 │   └─ worldstrat_x4_hat.yml
 ├─ scripts/
 │   ├─ make_lr_from_aid.py
 │   ├─ worldstrat_filter_align.py
 │   ├─ worldstrat_rgb_revisits_preview.py      # (추가) L2A 리비짓 RGB 진단/저장
 │   ├─ resize_dataset_pairs.py                 # (추가) HR 600 / LR 150 리사이즈 표준화
 │   ├─ train_hat.py
 │   ├─ train_ttst.py
 │   └─ train_srcnn.py
 ├─ data_cards/
 │   ├─ AID.md
 │   └─ WorldStrat.md
 ├─ reports/
 │   ├─ figures/
 │   └─ tables/
 ├─ env/
 │   ├─ rsisr.yaml
 │   └─ requirements.txt
 ├─ docs/            # (선택) mkdocs 사이트
 └─ logs/            # W&B run id / ckpt 링크
```

---

## 3) 데이터셋

### 3.1 AID (분류 데이터 → 합성 LR 생성)
- **출처**: https://github.com/YingdongKang/ACT_SR
- **다운샘플 규칙**: **x4, bicubic** (<!-- TODO: anti-alias on/off, 랜덤시드 고정값 -->)
- **Split**: <!-- TODO: train/val/test 수량 기입 -->

> 재현 스크립트: `scripts/make_lr_from_aid.py` (bicubic x4, 고정 시드)

### 3.2 WorldStrat (Pair: S2 10 m ↔ SPOT6/7 1.5 m)
- **출처**: https://www.kaggle.com/datasets/jucor1/worldstrat
- **이슈**: 센서/시간/구름/방사/정합 차이 → 전처리 필수

#### 3.2.1 파이프라인 단계 (요약)
1) **RGB 밴드만 사용**  
   - L2A 리비짓별 TIFF에서 (3,2,1) 채널을 선택해 RGB 구성 및 정규화/옵션 CLAHE.  
   - 구현: `scripts/worldstrat_rgb_revisits_preview.py` (아래 사용법 참조).

2) **리사이즈 표준화**  
   - **HR → 600×600**, **LR → 150×150** 고정.  
   - 구현: `scripts/resize_dataset_pairs.py` (아래 사용법 참조).

3) **정합/방사/난이도 필터** (OpenSR 영감)
   - **QA1(정합)**: LightGlue+DISK 특징점 매칭 기반, 변위 RMSE **≤ 0.75 LR px** (≈ 30 m×0.75).  
   - **QA2(방사)**: 밴드별 HM 적용 후 **RGB-SAD ≤ 5°**.  
   - **난이도**: HM/NoHM 각각에서 **SSIM Top-40%**(split별 분포 기준).

4) **데이터 변형/분리**  
   - train/val/test 분할 후 `<split>/{GT,LR}/<tile>.png` 구조로 저장.

> 주의: Sentinel-2 밴드 순서/폭은 제공본에 따라 상이할 수 있음. (3,2,1)이 실제 RGB에 해당하는지 **사전 확인** 필요.

#### 3.2.2 리비짓 진단/저장 스크립트 사용법
```bash
python scripts/worldstrat_rgb_revisits_preview.py \
  --dataset_path /PATH/worldstrat/dataset \
  --main_path    /PATH/worldstrat/dataset \
  --hr_rel hr_dataset/12bit \
  --lr_rel lr_dataset \
  --max_samples 5 \
  --apply_clahe 0
```
- 출력: `diagnostic_images_mini/`, `optimized_samples_mini/all_revisits/<AREA>/*.png`

#### 3.2.3 리사이즈 표준화 스크립트 사용법
```bash
# HR 600 / LR 150으로 리사이즈 (동일 파일명 기준 매칭 저장)
python scripts/resize_dataset_pairs.py \
  --src_hr_dir /PATH/WorldStrat/train/GT \
  --src_lr_dir /PATH/WorldStrat/train/LR \
  --dst_root   /PATH/WorldStrat_resized/train \
  --hr_size 600 --lr_size 150
```

---

## 4) 모델 & 방법

### 4.1 비교 모델
- **SRCNN / TTST / HAT**
  - 구현/커밋/체크포인트: <!-- TODO: 레포/커밋/ckpt 경로 -->
  - 공정성: 동일 다운샘플/스케일/스케줄/에폭/시드

### 4.2 실험 모듈
- **엣지 모듈**: (예) 입력 edge 채널 추가 또는 보조 loss(Edge-SSIM) <!-- TODO: 실제 적용 방식/하이퍼 -->
- **MAE 전이**: 사전학습 데이터/마스킹/패치/전이 방식 <!-- TODO -->

### 4.3 학습 세부
- **손실**: L1/L2 + Perceptual(VGG) + (선택) Edge/GAN
- **설정**: batch=<!-- TODO -->, epochs=<!-- TODO -->, lr=<!-- TODO -->, sched=<!-- TODO -->, AMP=on/off, EMA=on/off
- **시드/재현**: cudnn.deterministic=True, benchmark=False
- **환경 파일**: `env/rsisr.yaml` / `env/requirements.txt` (<!-- TODO: 채워서 커밋 -->)

---

## 5) 실험 (WorldStrat) – 설계 & 분석

### 5.1 데이터셋 구성 (OpenSR 영감)
- **Train**
  - **good 6k**: 5,925 쌍 (QA1/QA2/난이도 필터 통과)
  - **mix 6k**: 5,925 쌍 랜덤
  - **mix 20k**: 20,382 쌍 전체
- **Val**
  - **val_mixed**: 1,000 쌍 (전체 2,590 중 good:drop 비율(≈292:708) 반영)
  - **val_clean**: 456 쌍 (filtered)
  - **val_all**: 2,590 쌍 (전체)
- **평가 지표**: PSNR, SSIM

> 참고: https://esaopensr.github.io/opensr-test/ — QA1: LightGlue+DISK, QA2: HM 후 SAD≤5°.

### 5.2 Runpod 실험 설정
- **Loss**: L1  
- **GPU 0**
  - train: **mix 20k**; val: val_mixed, val_clean, val_all; batch=6
- **GPU 1 – (1)**
  - train: **good 6k**; val: 동일; batch=3
- **GPU 1 – (2)**
  - train: **mix 6k**; val: 동일; batch=3

### 5.3 핵심 결과 요약
- **mix 20k**(필터링 없는 all)에서 **PSNR·SSIM 모두 최고** → **데이터 수량** 효과.
- 다수 케이스에서 **mix 6k > good 6k**  
  - 이유: **텍스처 다양성 상실**(good은 밭/초지/수면 등 저주파 편중), **도심 엣지/고주파 노출 감소**.
- **인퍼런스 파이프라인**과 **학습 분포** 정렬이 중요:
  - **NoHM 데이터에 쓸 모델** ⇒ **NoHM로 학습**이 유리(PSNR·SSIM).
  - **HM 파이프라인(추론 시에도 HM 적용)** ⇒ **HM로 학습** 시 **SSIM 이득**. 단, **PSNR 목표**면 HM 단독 학습은 손해 가능.
- **목표 함수에 따라 전략 분기**:  
  - **SSIM 최적화** ⇒ HM 유리(특히 HM 검증셋).  
  - **PSNR 최적화** ⇒ NoHM가 전반적으로 유리.

> 상세 수치/그래프는 `reports/tables/*.csv`, `reports/figures/*.png` 참조.

---

## 6) 빠른 시작(Quickstart)

### 6.1 환경
```bash
# Conda (예시)
conda env create -f env/rsisr.yaml
conda activate rsisr
# 또는
pip install -r env/requirements.txt
```

### 6.2 데이터 준비
```text
DATA_ROOT/
 ├─ AID/...
 └─ WorldStrat/
     ├─ train/{GT,LR}
     ├─ val/{GT,LR}
     └─ test/{GT,LR}
```

### 6.3 전처리/필터 실행
```bash
# (선택) 리비짓 RGB 진단/저장
python scripts/worldstrat_rgb_revisits_preview.py \
  --dataset_path /PATH/worldstrat/dataset \
  --main_path    /PATH/worldstrat/dataset \
  --hr_rel hr_dataset/12bit \
  --lr_rel lr_dataset \
  --max_samples 5 \
  --apply_clahe 0

# (선택) 페어 리사이즈 표준화 (HR 600 / LR 150)
python scripts/resize_dataset_pairs.py \
  --src_hr_dir /PATH/WorldStrat/train/GT \
  --src_lr_dir /PATH/WorldStrat/train/LR \
  --dst_root   /PATH/WorldStrat_resized/train \
  --hr_size 600 --lr_size 150
```

### 6.4 학습 / 검증
```bash
# HAT (예시)
python scripts/train_hat.py -c configs/worldstrat_x4_hat.yml --wandb

# AID(x4) 공정 비교
python scripts/train_srcnn.py -c configs/aid_x4_bicubic.yml
python scripts/train_ttst.py  -c configs/aid_x4_bicubic.yml
python scripts/train_hat.py   -c configs/aid_x4_bicubic.yml
```

---

## 7) 평가 설정
- **지표**: PSNR↑, SSIM↑, (선택) LPIPS↓, DISTS↓, NIQE↓, MUSIQ↑, MANIQA↑
- **채널**: Y (기본) / RGB (보고 시 명시)
- **트랙**: **All** / **SSIM Top-40%** / **Strict-Align(ecc/shift/SAD)**

---

## 8) 결과 표 (자리표시자)
### 8.1 AID(x4) – 모델 비교 (RGB, test)
| Model | PSNR | SSIM | LPIPS | DISTS |
|------:|-----:|-----:|------:|------:|
| SRCNN |      |      |       |       |
| TTST  |      |      |       |       |
| HAT   |      |      |       |       |

### 8.2 WorldStrat(x4) – 전처리 트랙별 (HAT)
| Track | PSNR | SSIM | LPIPS | DISTS | 비고 |
|------|-----:|-----:|------:|------:|-----|
| All            | | | | |  |
| SSIM Top-40%   | | | | | 분포 40% 기준 |
| Strict-Align   | | | | | shift≤0.75px, SAD≤5° |

> 그래프/패널: `reports/figures/`

---

## 9) 한계 & 다음 과제
- 센서/시간차로 인한 잔차
- HM/NoHM 분포 편향
- 다음: 엣지×MAE 조합, 도메인 어댑테이션, 하드케이스 증강

---

## 10) 재현성
- 환경 파일: `env/rsisr.yaml`
- 주요 스크립트: `scripts/*.py`, 설정: `configs/*.yml`
- 실행 로그/체크포인트: <!-- TODO: W&B 프로젝트/엔티티/런 링크 또는 logs/ 경로 -->

---

## 11) 사이트(mkdocs) (선택)
```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs new .
# mkdocs.yml 구성 후
mkdocs serve
mkdocs gh-deploy
```

---

## 12) 인용 & 라이선스
- **데이터셋**
  - AID: https://github.com/YingdongKang/ACT_SR
  - WorldStrat: https://www.kaggle.com/datasets/jucor1/worldstrat
- **참고**: OpenSR-Test (QA1/QA2 워크플로 영감)
- **모델/코드베이스**: (HAT, TTST, 기타 참조 레포) <!-- TODO: 링크 기입 -->
- **라이선스**: <!-- TODO -->

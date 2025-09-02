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
- **파이프라인**
  1) **HM vs NoHM**: 히스토그램 매칭 적용 여부 비교
  2) **정합 품질**(기하):  
     - `shift_mag ≤ <!-- TODO: px (권장 1.0) -->`, `ecc_cc ≥ <!-- TODO: (권장 0.5) -->`
     - 특징점: **LightGlue + DISK**로 매칭 품질 기반 필터 (예: inlier 수, 매칭 점수 임계)
  3) **난이도 필터**: **SSIM Top-40%** (split별 분포 기준)
  4) **방사 일치**(스펙트럼 각): **RGB-SAD ≤ <!-- TODO: ° (권장 5°) -->**
- **폴더 규칙**: `<split>/{GT,LR}/<tile>.png` (GT 600×600 / LR 150×150)
- **필터 후 샘플 수**: <!-- TODO: train/val/test 남은 개수 기입 -->

> 상세 지침/근거는 `data_cards/WorldStrat.md`에 정리.

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

## 5) 빠른 시작(Quickstart)

### 5.1 환경
```bash
# Conda (예시)
conda env create -f env/rsisr.yaml
conda activate rsisr
# 또는
pip install -r env/requirements.txt
```

### 5.2 데이터 준비
```text
DATA_ROOT/
 ├─ AID/...
 └─ WorldStrat/
     ├─ train/{GT,LR}
     ├─ val/{GT,LR}
     └─ test/{GT,LR}
```

### 5.3 전처리/필터 실행
```bash
python scripts/worldstrat_filter_align.py   --data_root <DATA_ROOT>/WorldStrat   --hm <on|off>   --shift_thr <px> --ecc_thr <val>   --ssim_topk 0.40   --sad_thr <deg>   --out_dir data/WorldStrat_filtered
```

### 5.4 학습 / 검증
```bash
# HAT (예시)
python scripts/train_hat.py -c configs/worldstrat_x4_hat.yml --wandb

# AID(x4) 공정 비교
python scripts/train_srcnn.py -c configs/aid_x4_bicubic.yml
python scripts/train_ttst.py  -c configs/aid_x4_bicubic.yml
python scripts/train_hat.py   -c configs/aid_x4_bicubic.yml
```

---

## 6) 평가 설정
- **지표**: PSNR↑, SSIM↑, LPIPS↓, DISTS↓, (선택) NIQE↓, MUSIQ↑, MANIQA↑
- **채널**: Y (기본) / RGB (보고 시 명시)
- **트랙**: **All** / **SSIM Top-40%** / **Strict-Align(ecc/shift/SAD)**

---

## 7) 결과

### 7.1 AID(x4) – 모델 비교 (RGB, test)
| Model | PSNR | SSIM | LPIPS | DISTS |
|------:|-----:|-----:|------:|------:|
| SRCNN |      |      |       |       |
| TTST  |      |      |       |       |
| HAT   |      |      |       |       |

### 7.2 WorldStrat(x4) – 전처리 트랙별 (HAT)
| Track | PSNR | SSIM | LPIPS | DISTS | 비고 |
|------|-----:|-----:|------:|------:|-----|
| All            | | | | |  |
| SSIM Top-40%   | | | | | 분포 40% 기준 |
| Strict-Align   | | | | | shift≤…, ecc≥…, SAD≤… |

> 시각화는 `reports/figures/` 참고 (LR↑ vs SR vs GT, 경계 확대 포함).

---

## 8) 분석 & 논의
- 엣지 모듈 → 경계 품질/LPIPS 영향
- MAE 전이 → 저데이터 수렴/초기가속
- Top-40%만 학습 시 **편향**과 실제 일반화 간 균형

---

## 9) 한계 & 다음 과제
- 센서/시간차로 인한 잔차
- HM/NoHM 분포 편향
- 다음: 엣지×MAE 조합, 도메인 어댑트, 하드케이스 증강

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
- **모델/코드베이스**: (HAT, TTST, 기타 참조 레포) <!-- TODO: 링크 기입 -->
- **라이선스**: <!-- TODO -->

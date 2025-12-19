# Stage 2: Representation-Based XAI

## 개요

이 프로젝트는 **2-stage pipeline**으로 구성됩니다:

- **Stage 1**: Audio Deepfake Detector 학습 (이미 완료)
- **Stage 2**: XAI 모듈 학습 (Frozen detector 위에서)

## 아키텍처

```
┌─────────────────────────────────────┐
│  STAGE 1: Pretrained Detector      │
│  (avg_5_best.pth - FROZEN)          │
│  ├── SSL Encoder (XLSR)             │
│  └── ConformerTCM Classifier        │
└─────────────────────────────────────┘
            ↓ (SSL embeddings)
┌─────────────────────────────────────┐
│  STAGE 2: XAI Module (TRAINABLE)    │
│  ├── Importance Network             │
│  │   h(t) → w(t) ∈ [0,1]            │
│  └── Prototype Manager              │
│      ├── p_bonafide                 │
│      └── p_spoof                    │
└─────────────────────────────────────┘
            ↓
    Frame-level importance map
```

## 핵심 원리

### 1. Frozen Detector
- `avg_5_best.pth`에서 학습된 모델 로드
- **모든 파라미터 freeze** (gradient 없음)
- SSL embeddings만 추출

### 2. XAI Module (학습됨)
- **Importance Network**: 각 프레임의 중요도 w(t) 생성
- **Prototype Manager**: 클래스별 prototype 관리
  - EMA 모드 (default): Moving average로 업데이트
  - Fixed 모드 (ablation): 고정된 global mean
  - Learnable 모드 (fallback): Backprop으로 학습

### 3. Loss Function

#### Frame-level Contrastive Loss
```
ℓ_t = -log(exp(h(t)·p_y) / Σ_c exp(h(t)·p_c))
```

#### Importance-Weighted Objective
```
L_contrastive = Σ_t [w(t) * ℓ_t]
```

#### Regularization
- **Bonafide reg**: Bonafide 샘플은 낮은 importance
- **Temporal smoothness**: 시간적으로 부드러운 importance
- **Sparsity**: Spoof 샘플은 집중된 importance

## 사용법

### Stage 2 학습

```bash
cd /home/woongjae/ADD_LAB/Representation-grounded_XAI

# Stage 2 XAI 모듈 학습
bash script/train_xai_stage2.sh
```

### 설정 파일

`configs/xai_stage2.yaml`:

```yaml
model:
  name: "xai_stage2"

  # Pretrained detector (FROZEN)
  pretrained_checkpoint: "avg_5_best.pth"

  # XAI module (TRAINABLE)
  prototype_mode: "ema"  # 'ema', 'fixed', 'learnable'
  importance_hidden_dim: 256

  # Loss weights
  temperature: 0.07
  lambda_bonafide_reg: 0.1
  lambda_temporal_smooth: 0.1
  lambda_sparsity: 0.01
```

## Prototype 학습 모드

### EMA Mode (DEFAULT)
```python
prototype_mode: "ema"
ema_momentum: 0.99
```
- Exponential Moving Average로 prototype 업데이트
- 가장 안정적, **추천 설정**

### Fixed Mode (ABLATION)
```python
prototype_mode: "fixed"
```
- 전체 training set의 global mean으로 prototype 고정
- Ablation study용

### Learnable Mode (FALLBACK)
```python
prototype_mode: "learnable"
```
- Prototype이 learnable parameter
- EMA가 실패할 경우에만 사용

## 파일 구조

```
model/
  ├── xai_stage2.py          # Stage 2 XAI 모델
  ├── conformertcm.py         # Stage 1 detector (frozen)
  └── __init__.py

configs/
  └── xai_stage2.yaml         # Stage 2 설정

script/
  └── train_xai_stage2.sh     # Stage 2 학습 스크립트

avg_5_best.pth                # Stage 1 pretrained detector
```

## 중요 포인트

✅ **Stage 1은 완전히 frozen**
- 학습된 detector의 파라미터는 전혀 변경되지 않음
- SSL embeddings만 추출에 사용

✅ **Stage 2만 학습됨**
- Importance Network
- Prototypes (EMA 모드에서는 gradient 없이 moving average로 업데이트)

✅ **Detector score는 사용 안 함**
- Stage 1의 classification score는 supervision으로 사용되지 않음
- Representation space에서 직접 explanation 학습

✅ **Frame-level supervision**
- Utterance-level label만 있지만
- Frame-level contrastive loss로 학습

## 결과 해석

### Bonafide 샘플
- 낮은 total importance
- 균일한 importance 분포
- 부드러운 시간적 변화

### Spoof 샘플
- 높은 total importance
- Artifact 영역에 집중된 importance
- 해당 프레임에서 높은 spoof prototype similarity

## 문제 해결

### Issue: Loss가 감소하지 않음
- Learning rate 줄이기 (1e-5)
- Regularization weight 조정
- Prototype mode를 'fixed'로 변경해보기

### Issue: Importance가 모두 0 또는 1
- Regularization weight 조정
- Temperature 변경 (0.05 ~ 0.1)

### Issue: OOM (Out of Memory)
- Batch size 줄이기 (24 → 16)
- Gradient accumulation 사용

## Citation

```bibtex
@inproceedings{xai_stage2_2025,
  title={Representation-Based XAI for Audio Deepfake Detection},
  year={2025}
}
```

# 2-Stage XAI Pipeline 사용 가이드

## 🚀 빠른 시작

### 전체 파이프라인 실행 (Stage 1 + 2)

```bash
cd /home/woongjae/ADD_LAB/Representation-grounded_XAI

# 전체 파이프라인 실행
bash script/train_full_pipeline.sh
```

이 명령어는:
1. **Stage 1** 학습 → `out/stage1_detector.pth` 저장
2. 자동으로 **Stage 2** 학습 → Stage 1 모델 로드 → `out/stage2_xai.pth` 저장

---

## 📋 개별 Stage 실행

### Stage 1만 실행

```bash
bash script/train_full_pipeline.sh --stage1
```

- Detector만 학습
- `out/stage1_detector.pth`에 저장

### Stage 2만 실행

```bash
bash script/train_full_pipeline.sh --stage2
```

- XAI 모듈만 학습
- 기존 Stage 1 모델 필요:
  - `out/stage1_detector.pth` (우선)
  - `avg_5_best.pth` (fallback)

---

## 🔧 커스터마이징

### 모델 경로 변경

`script/train_full_pipeline.sh` 수정:

```bash
# Line 26-27 수정
STAGE1_MODEL="${OUT_DIR}/my_detector.pth"
STAGE2_MODEL="${OUT_DIR}/my_xai.pth"
```

### Hyperparameter 조정

**Stage 1** (`configs/conformertcm_baseline.yaml`):
```yaml
model:
  emb_size: 144        # Conformer embedding size
  num_encoders: 4      # Conformer layers
  heads: 4             # Attention heads
  kernel_size: 31      # Convolution kernel
```

**Stage 2** (`configs/xai_stage2.yaml`):
```yaml
model:
  prototype_mode: "ema"           # 'ema', 'fixed', 'learnable'
  importance_hidden_dim: 256      # Importance network size

  # Loss weights
  temperature: 0.07
  lambda_bonafide_reg: 0.1
  lambda_temporal_smooth: 0.1
  lambda_sparsity: 0.01
```

### Batch Size / Epochs 조정

`script/train_full_pipeline.sh` 수정:

```bash
# Stage 1 (line 57-58)
--batch_size 32 \
--num_epochs 30 \

# Stage 2 (line 137-138)
--batch_size 24 \
--num_epochs 50 \
```

---

## 📊 실행 흐름

```
┌──────────────────────────────────────┐
│  train_full_pipeline.sh              │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│  STAGE 1: Detector Training          │
│  - Config: conformertcm_baseline.yaml│
│  - Train: SSL + Conformer            │
│  - Save: stage1_detector.pth         │
└──────────────────────────────────────┘
              ↓ (자동 연결)
┌──────────────────────────────────────┐
│  STAGE 2: XAI Training                │
│  - Config: xai_stage2.yaml           │
│  - Load: stage1_detector.pth (freeze)│
│  - Train: Importance + Prototypes    │
│  - Save: stage2_xai.pth              │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│  Complete! Both models in out/       │
└──────────────────────────────────────┘
```

---

## 🎯 사용 예시

### 예시 1: 처음부터 전체 학습

```bash
# Stage 1 + Stage 2 모두 학습
bash script/train_full_pipeline.sh

# 결과:
# ✓ out/stage1_detector.pth (Detector)
# ✓ out/stage2_xai.pth (XAI)
```

### 예시 2: Stage 1만 먼저 학습

```bash
# Stage 1만 학습
bash script/train_full_pipeline.sh --stage1

# 결과 확인 후 Stage 2 학습
bash script/train_full_pipeline.sh --stage2
```

### 예시 3: 기존 모델로 Stage 2만 학습

```bash
# avg_5_best.pth가 이미 있는 경우
bash script/train_full_pipeline.sh --stage2

# 자동으로 avg_5_best.pth 로드
```

### 예시 4: Prototype 모드 변경

```bash
# 1. xai_stage2.yaml 수정
# prototype_mode: "fixed"  # EMA → Fixed로 변경

# 2. Stage 2만 재학습
bash script/train_full_pipeline.sh --stage2
```

---

## 📁 출력 파일 구조

```
out/
├── stage1_detector.pth      # Stage 1 trained model
├── stage2_xai.pth            # Stage 2 XAI module
└── xai_stage2_temp.yaml      # 임시 config (자동 삭제됨)

logs/
├── stage1_detector/          # Stage 1 tensorboard logs
│   └── events.out.tfevents.*
└── stage2_xai/               # Stage 2 tensorboard logs
    └── events.out.tfevents.*
```

---

## 🐛 트러블슈팅

### Issue: "No Stage 1 model found"

**원인**: Stage 2를 실행했는데 Stage 1 모델이 없음

**해결**:
```bash
# Option 1: Stage 1 먼저 학습
bash script/train_full_pipeline.sh --stage1

# Option 2: 기존 모델을 out/ 폴더로 복사
cp avg_5_best.pth out/stage1_detector.pth
```

### Issue: "STAGE 1 FAILED"

**원인**: Stage 1 학습 중 오류

**해결**:
1. GPU 메모리 확인: `nvidia-smi`
2. Batch size 줄이기 (32 → 16)
3. Config 파일 경로 확인

### Issue: "STAGE 2 FAILED"

**원인**: Stage 2 학습 중 오류

**해결**:
1. Stage 1 모델 로드 확인
2. Config 파일에서 `pretrained_checkpoint` 경로 확인
3. Batch size 줄이기 (24 → 16)

### Issue: OOM (Out of Memory)

**해결**:
```bash
# train_full_pipeline.sh 수정
--batch_size 16   # Stage 1: 32 → 16
--batch_size 12   # Stage 2: 24 → 12
```

---

## 📈 학습 모니터링

### TensorBoard

```bash
# Stage 1 로그 확인
tensorboard --logdir logs/stage1_detector

# Stage 2 로그 확인
tensorboard --logdir logs/stage2_xai

# 모든 로그 확인
tensorboard --logdir logs/
```

### 학습 중 로그 확인

```bash
# 실행 중인 스크립트의 출력 확인
tail -f nohup.out  # nohup으로 실행한 경우
```

---

## ⚙️ 고급 사용법

### 병렬 실험 실행

```bash
# 서로 다른 설정으로 여러 실험 병렬 실행

# Terminal 1: EMA mode
bash script/train_full_pipeline.sh --stage2

# Terminal 2: Fixed mode (config 수정 후)
bash script/train_full_pipeline.sh --stage2

# Terminal 3: Learnable mode (config 수정 후)
bash script/train_full_pipeline.sh --stage2
```

### 다른 데이터셋으로 학습

`train_full_pipeline.sh` 수정:

```bash
# Line 18 수정
DATABASE_PATH="/path/to/your/dataset"
PROTOCOL_PATH="/path/to/your/protocol.txt"
```

---

## 🎓 Best Practices

1. **Stage 1 먼저 완벽히 학습**
   - EER < 1% 정도 목표
   - 충분한 epoch (30-50)

2. **Stage 2는 빠르게 수렴**
   - 보통 10-20 epoch면 충분
   - Early stopping 활용

3. **Ablation Study**
   - EMA, Fixed, Learnable 모드 모두 테스트
   - Loss weight 변경해가며 실험

4. **리소스 관리**
   - Stage 1: 더 큰 batch size (32)
   - Stage 2: 작은 batch size (24)
   - GPU 메모리 부족시 줄이기

---

## 📚 추가 문서

- **Stage 2 상세 설명**: `README_STAGE2_XAI.md`
- **모델 아키텍처**: `model/xai_stage2.py` 주석 참고
- **Config 옵션**: `configs/xai_stage2.yaml` 주석 참고

---

## 🤝 문의

문제가 발생하면:
1. 로그 파일 확인
2. Config 파일 재확인
3. GPU 메모리 상태 확인 (`nvidia-smi`)

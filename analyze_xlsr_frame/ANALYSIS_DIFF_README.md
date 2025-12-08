# Temporal Difference Learning Analysis Pipeline

Complete implementation of Figure 1 and Figure 2 from the paper:
**"Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection"**

## 📋 Overview

This pipeline analyzes ASVspoof 2019 LA dataset using temporal difference learning:

- **Figure 1**: Raw frame-to-frame differences
- **Figure 2**: Cosine similarity between direction vectors (Equation 1 & 2)

## 📂 Files

```
Representation-grounded_XAI/
├── asvspoof_protocol_parser.py          # Parse ASVspoof protocol
├── extract_embeddings.py                # Extract & save SSL embeddings
├── figure1_raw_difference_analysis.py   # Figure 1 analysis
├── figure2_cosine_similarity_analysis.py # Figure 2 analysis (Eq 1 & 2)
├── run_full_analysis.py                 # Complete pipeline
├── temporal_difference_learning.py      # Core math (Eq 1 & 2)
└── model/conformertcm.py               # SSL model (wav2vec2-XLSR)
```

## 🚀 Quick Start

### Complete Pipeline (Recommended)

Run everything in one command:

```bash
python run_full_analysis.py \
    --protocol_path /home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
    --audio_base_dir /path/to/ASVspoof2019_LA_train/flac/ \
    --extract_embeddings \
    --embedding_dir /nvme3/wj/embeddings/ \
    --output_dir ./results/ \
    --device cuda \
    --batch_size 16
```

**Output:**
```
results/
├── figure1_results.pkl
├── figure1_raw_difference_boxplot.png
├── figure1_raw_difference_boxplot.pdf
├── figure2_results.pkl
├── figure2_cosine_similarity_boxplots.png
├── figure2_cosine_similarity_boxplots.pdf
├── figure2_mean_boxplot.png
└── figure2_std_boxplot.png
```

---

## 📝 Step-by-Step Usage

### Step 1: Extract Embeddings (One-Time)

Extract SSL embeddings and save to `/nvme3/wj/embeddings/`:

```bash
python extract_embeddings.py \
    --protocol_path /home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
    --audio_base_dir /path/to/ASVspoof2019_LA_train/flac/ \
    --output_dir /nvme3/wj/embeddings/ \
    --device cuda \
    --batch_size 16
```

**What it does:**
- Loads audio files in batches
- Extracts frame-level embeddings using wav2vec2-XLSR (1024-dim)
- Saves each embedding as `.pt` file
- Creates directory structure:
  ```
  /nvme3/wj/embeddings/
  ├── bonafide/
  │   ├── LA_T_1138215.pt
  │   ├── LA_T_1271820.pt
  │   └── ...
  ├── A01/
  │   ├── LA_T_1004644.pt
  │   └── ...
  ├── A02/
  ...
  └── embedding_index.pkl
  ```

**Time estimate:** ~2-4 hours for 25,000 files (depends on GPU)

**Resume:** If interrupted, re-run same command. It will skip already processed files.

---

### Step 2: Figure 1 Analysis

Compute raw frame-to-frame differences:

```bash
python figure1_raw_difference_analysis.py \
    --embedding_dir /nvme3/wj/embeddings/ \
    --output_dir ./results/
```

**Formula:**
```
For each utterance:
    d_t = x_{t+1} - x_t  (raw difference, NO normalization)
    utter_mean = mean(d_t) over all dimensions and frames
```

**Output:**
- `figure1_results.pkl`: Dictionary of results
- `figure1_raw_difference_boxplot.png`: Boxplot (PNG)
- `figure1_raw_difference_boxplot.pdf`: Boxplot (PDF)

**Time estimate:** ~1-2 minutes

---

### Step 3: Figure 2 Analysis

Compute cosine similarity using Equation 1 & 2:

```bash
python figure2_cosine_similarity_analysis.py \
    --embedding_dir /nvme3/wj/embeddings/ \
    --output_dir ./results/
```

**Formulas:**
```
Equation (1): Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||
Equation (2): cos(θ_t) = Δx_t · Δx_{t+1}

For each utterance:
    μ = mean(cos(θ_t))
    σ = std(cos(θ_t))
```

**Output:**
- `figure2_results.pkl`: Dictionary of mean and std results
- `figure2_cosine_similarity_boxplots.png`: Combined boxplots (PNG)
- `figure2_cosine_similarity_boxplots.pdf`: Combined boxplots (PDF)
- `figure2_mean_boxplot.png`: Mean only (PNG)
- `figure2_std_boxplot.png`: Std only (PNG)

**Time estimate:** ~2-3 minutes

---

## 🔧 Advanced Options

### Skip Embedding Extraction

If embeddings already extracted, run analysis only:

```bash
python run_full_analysis.py \
    --protocol_path /path/to/protocol.txt \
    --embedding_dir /nvme3/wj/embeddings/ \
    --output_dir ./results/
```

### Run Only Figure 1

```bash
python run_full_analysis.py \
    --protocol_path /path/to/protocol.txt \
    --embedding_dir /nvme3/wj/embeddings/ \
    --output_dir ./results/ \
    --skip_figure2
```

### Run Only Figure 2

```bash
python run_full_analysis.py \
    --protocol_path /path/to/protocol.txt \
    --embedding_dir /nvme3/wj/embeddings/ \
    --output_dir ./results/ \
    --skip_figure1
```

### Parse Protocol Only

```bash
python asvspoof_protocol_parser.py \
    --protocol_path /path/to/protocol.txt \
    --base_dir /path/to/flac/
```

---

## 📊 Expected Results

### Figure 1: Raw Frame Difference

**Typical values:**
- Bonafide: Lower mean raw difference
- Spoof attacks: Higher mean raw difference (especially certain attacks)

### Figure 2: Cosine Similarity

**Typical values:**

**Mean (μ):**
- Bonafide: Higher mean cosine similarity (~0.8-0.9)
  - Smoother trajectory, similar directions
- Spoof: Lower mean cosine similarity (~0.5-0.7)
  - More abrupt changes, different directions

**Std (σ):**
- Bonafide: Lower std deviation (~0.1-0.2)
  - Consistent temporal patterns
- Spoof: Higher std deviation (~0.2-0.4)
  - Inconsistent, variable temporal patterns

---

## 🔍 Understanding the Output

### Embedding Files

Each `.pt` file contains:
```python
embedding = torch.load("LA_T_1138215.pt")
print(embedding.shape)  # (T, 1024)
# T = number of frames (varies by audio length)
# 1024 = wav2vec2-XLSR output dimension
```

### Result Files

**figure1_results.pkl:**
```python
import pickle
with open("figure1_results.pkl", "rb") as f:
    results = pickle.load(f)

print(results.keys())
# dict_keys(['bonafide', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06'])

print(len(results['bonafide']))  # Number of bonafide utterances
print(results['bonafide'][:5])   # First 5 utter_mean values
```

**figure2_results.pkl:**
```python
import pickle
with open("figure2_results.pkl", "rb") as f:
    results = pickle.load(f)

mean_results = results['mean']
std_results = results['std']

print(mean_results['bonafide'][:5])  # First 5 μ values
print(std_results['bonafide'][:5])   # First 5 σ values
```

---

## 💾 Disk Space Requirements

**Embeddings:** ~15-20 GB for full ASVspoof 2019 LA train set
- ~600 KB per file (T=150 frames average)
- 25,000 files total

**Results:** ~10-50 MB
- Pickle files with raw results
- PNG/PDF plots

**Recommendation:** Use `/nvme3/wj/` for embeddings (fast SSD)

---

## ⚡ Performance Tips

1. **Batch Size:**
   - GPU memory: 16GB → batch_size=16
   - GPU memory: 24GB → batch_size=32
   - GPU memory: 48GB → batch_size=64

2. **Resume Mode:**
   - Always enabled by default
   - Safe to interrupt and restart

3. **Parallel Processing:**
   - Analysis scripts are single-threaded
   - Can run Figure 1 and Figure 2 in parallel after embedding extraction

---

## 🐛 Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
python extract_embeddings.py ... --batch_size 8
```

### Missing Audio Files

Check base directory path:
```bash
ls /path/to/ASVspoof2019_LA_train/flac/LA_T_*.flac | head
```

### Protocol Parse Error

Verify protocol file format:
```bash
head -n 5 /path/to/ASVspoof2019.LA.cm.train.trn.txt
```

Expected format:
```
LA_0079 LA_T_1138215 - - bonafide
LA_0083 LA_T_9228662 - A02 spoof
```

---

## 📚 Implementation Details

### SSL Model

- **Model:** wav2vec2-XLSR 300M
- **Path:** `/home/woongjae/wildspoof/xlsr2_300m.pt`
- **Framework:** fairseq
- **Output:** 1024-dimensional frame-level embeddings

### Equations

**Equation (1) - Direction Vectors:**
```python
Δx_t = (x_{t+1} - x_t) / ||x_{t+1} - x_t||
```

**Equation (2) - Cosine Similarity:**
```python
cos(θ_t) = (Δx_t · Δx_{t+1}) / (||Δx_t|| * ||Δx_{t+1}||)
```

Since Δx_t is normalized, this simplifies to:
```python
cos(θ_t) = Δx_t · Δx_{t+1}
```

### Attack Types

- **bonafide:** Real human speech
- **A01-A06:** Different synthesis/conversion attacks
  - A01: TTS (Text-to-Speech)
  - A02: TTS
  - A03: TTS
  - A04: TTS
  - A05: VC (Voice Conversion)
  - A06: VC

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@inproceedings{tdam2024,
  title={Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection},
  author={...},
  booktitle={...},
  year={2024}
}
```

---

## ✅ Checklist

Before running:
- [ ] Protocol file exists
- [ ] Audio files accessible
- [ ] `/nvme3/wj/` has enough space (~20 GB)
- [ ] CUDA available (`torch.cuda.is_available()`)
- [ ] fairseq installed
- [ ] SSL model exists at `/home/woongjae/wildspoof/xlsr2_300m.pt`

---

**Ready to run!** 🚀

For questions or issues, check the code comments or run with `--help` flag.

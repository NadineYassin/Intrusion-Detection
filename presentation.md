---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #004654ff;
  }
  h2 {
    color: #003742ff;
  }
  code {
    background: #16213e;
  }
  table {
    font-size: 0.8em;
  }
  strong {
    color: #002128ff;
  }
---

# Network Intrusion Detection System
## Using Machine Learning to Detect Cyber Attacks

---

# The Problem

**Networks are constantly under attack.**

- Hackers try to steal data, break systems, or shut servers down
- Normal firewalls use simple rules and can be tricked

**Our Solution:** Train ML models to automatically detect 6 types of attacks

---

# Related Work & Our Contributions

![w:400](paper.png)

| Paper | Our Work |
|-------|----------|
| Feature selection (Info Gain) | All 52 features |
| Single model | **RF vs XGBoost** |
| No imbalance handling | **SMOTE + Undersampling** |
| No resource tracking | **Memory & CPU monitoring** |
| No demo | **Streamlit app** |

---

# The Dataset: CICIDS2017

Created by **Canadian Institute for Cybersecurity** (University of New Brunswick)

| Attack Type | Samples | Description |
|------------|---------|-------------|
| Normal Traffic | 2,095,057 | Regular browsing, email, etc. |
| DoS | 193,745 | Flood server to crash it |
| DDoS | 128,014 | DoS from multiple sources |
| Port Scanning | 90,694 | Find open ports (reconnaissance) |
| Brute Force | 9,150 | Guess passwords repeatedly |
| Web Attacks | 2,143 | SQL injection, XSS |
| Bots | 1,948 | Remote-controlled malware |

---

# How It Works

```
Network Traffic → Packet Capture → Feature Extraction → ML Model → Attack/Normal?
   (live data)      (Wireshark)     (CICFlowMeter)      (XGBoost)
```

**52 Features** extracted from each network connection:
- **Packet info:** sizes, counts, lengths
- **Timing:** duration, inter-arrival times
- **Speed:** bytes/sec, packets/sec
- **TCP Flags:** FIN, SYN, ACK counts

---

# What We Did: Data Preprocessing

### 1. Loaded the Data
- **2.8 million+** network flow records
- Each record has **52 features** + attack label

### 2. Train/Test Split
- **70% training**, **30% testing**
- Stratified split (keeps attack ratios balanced)

### 3. Feature Scaling
- Applied **RobustScaler** (handles outliers well)

---

# What We Did: Handling Imbalanced Data

**Problem:** Normal traffic dominates (2M vs 1.9K bots)

| Technique | What We Did |
|-----------|-------------|
| **Undersampling** | Reduced Normal Traffic: 2M → 500K |
| **SMOTE** | Generated synthetic minority samples |

**Then trained:** Random Forest (200 trees) & XGBoost (150 estimators)

---

# Model 1: Random Forest

**Hyperparameters:**
- 200 trees, max_depth=None, min_samples_split=5

**Results:**

| Metric | Score |
|--------|-------|
| Accuracy | 99.88% |
| Precision | 99.96% |
| Recall | 99.88% |
| F1 Score | 99.92% |

**Resources:** 1614 MB memory, 106 sec training

---

# Model 2: XGBoost

**Hyperparameters:**
- 150 estimators, max_depth=3, learning_rate=0.3

**Results:**

| Metric | Score |
|--------|-------|
| Accuracy | 99.90% |
| Precision | 99.91% |
| Recall | **99.90%** |
| F1 Score | 99.90% |

**Resources:** 1123 MB memory, **27 sec training**

---

# Comparison: Which is Better?

| Metric | Random Forest | XGBoost | Winner |
|--------|--------------|---------|--------|
| Accuracy | 99.88% | **99.90%** | XGBoost |
| Recall | 99.88% | **99.90%** | XGBoost |
| Memory | 1614 MB | **1123 MB** | XGBoost |
| Training Time | 106 sec | **27 sec** | XGBoost |

### Winner: **XGBoost**
- Higher recall (catches more attacks)
- 4x faster training
- 30% less memory

---

# Per-Attack Performance (XGBoost)

| Attack Type | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| Normal Traffic | 1.00 | 1.00 | 1.00 |
| DoS | 1.00 | 1.00 | 1.00 |
| DDoS | 1.00 | 1.00 | 1.00 |
| Port Scanning | 0.99 | 1.00 | 0.99 |
| Brute Force | 1.00 | 1.00 | 1.00 |
| Web Attacks | 0.99 | 0.99 | 0.99 |
| Bots | 0.69 | 0.95 | 0.80 |

**Note:** Bots hardest to detect (fewest samples + similar to normal traffic)

---

# Conclusion

**We built an ML-based Network Intrusion Detection System that:**

- Detects **6 attack types** with **99.9% accuracy**
- Uses **52 network flow features**
- **XGBoost** outperforms Random Forest in speed & accuracy
- Deployed as interactive **Streamlit demo**

### Future Work
- Add more attack types
- Real-time PCAP processing
- Deploy on actual network

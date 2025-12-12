# ANN Final Project

# Bilingual Hate Speech Detection using BiLSTM

A deep learning project for detecting hate speech in Filipino and English text using Bidirectional LSTM neural networks.

## 📋 Project Overview

This project implements a hate speech detection system trained on 80,867 bilingual (Filipino + English) samples using a BiLSTM architecture. The system includes hyperparameter tuning across 5 different configurations and a Facebook-style GUI application for real-time detection.

## 🎯 Features

- **Bilingual Support**: Detects hate speech in both Filipino and English
- **BiLSTM Architecture**: 2-3 layer Bidirectional LSTM with 128-256 hidden units
- **Hyperparameter Tuning**: 5 configurations tested with comprehensive comparison
- **Interactive GUI**: Facebook-style social media interface with real-time hate speech filtering
- **High Performance**: 70.53% validation accuracy, 67.57% F1-score

## 📊 Dataset

- **Total Samples**: 80,867
- **Languages**: Filipino (43.1%) + English (56.9%)
- **Sources**: 
  - Filipino Twitter hate speech
  - Filipino TikTok hate speech
  - English cyberbullying tweets
- **Split**: 80% train, 10% validation, 10% test
- **Class Distribution**: 67.17% hate speech, 32.83% non-hate

## 🏗️ Model Architecture

```
BiLSTM Hate Speech Classifier
├── Embedding Layer (vocab_size × 128)
├── Bidirectional LSTM (2-3 layers, 64-256 hidden units)
├── Dropout (0.2-0.5)
├── Fully Connected Layers (256 → 64 → 1)
└── Sigmoid Activation
```

**Parameters**: 862K - 7.8M (depending on configuration)

## 🔬 Hyperparameter Tuning Results

### Configuration Comparison

| Config | Accuracy | F1-Score | Precision | Recall | Time | Parameters |
|--------|----------|----------|-----------|--------|------|------------|
| **Config 4** ⭐ | **70.53%** | 66.50% | 67.28% | 65.67% | 7.3 min | 3.7M |
| Config 1 | 70.42% | 64.26% | 68.63% | 60.42% | 4.5 min | 3.2M |
| Config 3 | 70.39% | **67.57%** ⭐ | 65.72% | **69.53%** ⭐ | **4.2 min** ⚡ | 862K |
| Config 2 | 69.49% | 64.79% | 66.56% | 63.11% | 17.9 min | 7.8M |
| Config 5 ❌ | 54.32% | 0.00% | 0.00% | 0.00% | 4.0 min | 3.2M |

### Configuration Details

#### **Config 1: Baseline Model**
- Learning Rate: 0.001, Batch Size: 64, Hidden: 128, Dropout: 0.3
- Optimizer: Adam, Epochs: 10, Layers: 2
- **Best for**: Balanced performance

#### **Config 2: Extended Training**
- Learning Rate: 0.0005, Batch Size: 32, Hidden: 256, Dropout: 0.4
- Optimizer: Adam, Epochs: 15, Layers: 2
- **Best for**: Larger model capacity

#### **Config 3: Fast Training**
- Learning Rate: 0.01, Batch Size: 128, Hidden: 64, Dropout: 0.2
- Optimizer: RMSprop, Epochs: 20, Layers: 2
- **Best for**: Quick prototyping, highest F1-score

#### **Config 4: Deep Model** ⭐
- Learning Rate: 0.0008, Batch Size: 64, Hidden: 128, Dropout: 0.35
- Optimizer: Adam, Epochs: 12, Layers: 3
- **Best for**: Maximum accuracy, production use

#### **Config 5: High Regularization** ❌
- Learning Rate: 0.001, Batch Size: 64, Hidden: 128, Dropout: 0.5
- Optimizer: SGD, Epochs: 10, Layers: 2
- **Result**: Failed - over-regularization prevented learning

## 🔑 Key Findings

### ✅ What Works:
- **Adam optimizer** with learning rate 0.0008-0.001
- **Dropout** between 0.2-0.35
- **10-12 epochs** sufficient (more doesn't help)
- **3 LSTM layers** give marginal improvement (+0.11%)

### ❌ What Doesn't Work:
- **High dropout (0.5)** - prevents learning completely
- **SGD optimizer** without proper tuning
- **Training beyond 15 epochs** - no benefit, just wastes time

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Steven-zx/ANN-Final-Project.git
cd ANN-Final-Project

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch pandas numpy scikit-learn tqdm
```

## 📝 Usage

### Train the Model

```bash
# Train baseline model
python train_bilstm.py

# Run full hyperparameter tuning (5 configs)
python hyperparameter_tuning.py

# Train only new configs (4 & 5)
python train_new_configs.py
```

### Test the Model

```bash
# Test threshold optimization
python find_threshold.py

# Quick model test
python test_model.py
```

### Run GUI Application

```bash
# Launch Facebook-style social media app
python social_media_app.py
```

The GUI features:
- Create posts with hate speech detection
- Real-time analysis with loading animation
- Posts blocked if hate speech detected
- Clean posts published to feed

## 📂 Project Structure

```
ANN-Final-Project/
├── train_bilstm.py              # Main training script
├── hyperparameter_tuning.py     # Full hyperparameter tuning
├── train_new_configs.py         # Train configs 4 & 5 only
├── social_media_app.py          # Facebook-style GUI
├── rnn_model.py                 # BiLSTM model architecture
├── text_preprocessing.py        # Text preprocessing & vocabulary
├── load_unified_dataset.py      # Dataset loader
├── find_threshold.py            # Threshold optimization
├── test_model.py                # Model testing
├── compare_all_configs.py       # Generate comparison report
│
├── hatespeech/                  # Main dataset
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
│
├── filipino-tiktok-hatespeech-main/  # TikTok dataset
├── Models/                      # Trained model checkpoints
│   ├── best_bilstm_model.pt
│   ├── model_config1_baseline.pt
│   ├── model_config2_extended.pt
│   ├── model_config3_fast.pt
│   ├── model_config4_deep.pt
│   └── model_config5_regularized.pt
│
└── vocabulary.pkl               # Saved vocabulary (20K words)
```

## 🎓 Course Information

- **Course**: CCS 248 - Artificial Neural Networks
- **Project**: Final Project - Hate Speech Detection
- **Date**: December 2025
- **Requirements**: 
  - RNN-based architecture (BiLSTM)
  - Hyperparameter tuning (5 configurations)
  - GUI application
  - Comprehensive evaluation

## 📈 Performance Metrics

### Best Model (Config 4):
- **Validation Accuracy**: 70.53%
- **F1-Score**: 66.50%
- **Precision**: 67.28%
- **Recall**: 65.67%
- **Training Time**: 7.3 minutes
- **Detection Threshold**: 0.8 (optimized to reduce false positives)

### Test Examples:
```python
✅ "you are the best thing in the world" → SAFE (8% hate probability)
✅ "tang ina mo gago ka" → HATE (100% hate probability)
✅ "have a great day" → SAFE (15.5% hate probability)
❌ "thank you so much for your help" → FALSE POSITIVE (86% hate probability)
```

## 🔧 Technologies Used

- **PyTorch** 2.9.1 - Deep learning framework
- **Python** 3.12
- **Tkinter** - GUI framework
- **pandas** - Data manipulation
- **scikit-learn** - Metrics and evaluation
- **tqdm** - Progress bars

## 📊 Files Generated

After training, the following files are created:

- `best_bilstm_model.pt` - Best model checkpoint
- `vocabulary.pkl` - Vocabulary (20K words)
- `hyperparameter_comparison_*.csv` - Config comparison table
- `detailed_results_*.json` - Full training history
- `tuning_summary_*.txt` - Summary report
- `complete_comparison_5configs_*.csv` - Final comparison
- `complete_analysis_*.txt` - Detailed analysis

## 🤝 Contributing

This is a course project. For educational purposes only.

## 📄 License

Educational use only - ANN Course Final Project

## 👥 Authors

- Steven-zx

## 🙏 Acknowledgments

- Filipino Text Benchmarks repository for model architecture inspiration
- HuggingFace datasets for Filipino hate speech data
- Course instructor for guidance and requirements

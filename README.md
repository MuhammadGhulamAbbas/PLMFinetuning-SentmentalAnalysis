# 💬 PLM Fine-Tuning for Sentiment Analysis

This repository provides a comparative study of **Classical Machine Learning methods** and **Pre-trained Language Models (PLMs)** for sentiment analysis. It includes fine-tuning experiments using **LoRA (Low-Rank Adaptation)** on **DistilBERT** and **RoBERTa**, and benchmarks them against classical algorithms like **Logistic Regression** and **Naive Bayes**.  

📌 **Focus**: Analyze accuracy, performance, and resource efficiency for various models on sentiment classification tasks.

📁 **Contributor**:  
**Muhammad Ghulam Abbas (29417)**

---

## 📂 Repository Contents

| Notebook | Description |
|----------|-------------|
| [`Classical-Sentiment-Analysis.ipynb`](https://github.com/MuhammadGhulamAbbas/PLMFinetuning-SentmentalAnalysis/blob/main/Classical-Sentiment-Analysis.ipynb) | Implements classical ML models: Naive Bayes, Logistic Regression, k-NN, and Random Forest with preprocessing & TF-IDF vectorization |
| [`distilbert-sentiment-analysis.ipynb`](https://github.com/MuhammadGhulamAbbas/PLMFinetuning-SentmentalAnalysis/blob/main/distilbert-sentiment-analysis.ipynb) | Fine-tunes DistilBERT using LoRA across multiple configurations |
| [`roberta-sentiment-lora-finetune.ipynb`](https://github.com/MuhammadGhulamAbbas/PLMFinetuning-SentmentalAnalysis/blob/main/roberta-sentiment-lora-finetune.ipynb) | Fine-tunes RoBERTa with LoRA and evaluates its effectiveness |
| [`benchmarking-sentiment-transformers.ipynb`](https://github.com/MuhammadGhulamAbbas/PLMFinetuning-SentmentalAnalysis/blob/main/benchmarking-sentiment-transformers.ipynb) | Evaluates Hugging Face's benchmark DistilBERT sentiment model (`distilbert-base-uncased-finetuned-sst-2-english`) |

---

## 📊 Model Overview

### Classical Machine Learning
- ✅ **Best Model**: Logistic Regression – **89.27% Accuracy**
- Other Models: Naive Bayes, k-NN, Random Forest
- Techniques: Text preprocessing, TF-IDF, CountVectorizer

### Fine-Tuned PLMs (with LoRA)
- 🔍 **DistilBERT**
  - Best Accuracy: **87.15%**
  - Best Config: `LoRA Rank: 4`, `Alpha: 8`, `Dropout: 0.10`
- 🔍 **RoBERTa**
  - Accuracy: **78.00%**
  - Performed lower due to limited training settings

### Benchmark Transformer Model
- 🤖 **Hugging Face DistilBERT (SST-2)**:  
  - Accuracy: **90%**
  - Best overall performance
  - Resource-intensive (GPU required)

---

## 📈 Performance Summary

| Model            | Accuracy | Training Time | Resource Needs |
|------------------|----------|----------------|----------------|
| Naive Bayes      | 87.98%   | Low            | Minimal        |
| Logistic Regression | **89.27%** | Moderate     | Minimal        |
| k-NN             | 79.45%   | Moderate       | Moderate       |
| Random Forest    | 85.49%   | High           | Moderate       |
| DistilBERT (LoRA) | 87.15%  | Very High      | GPU-intensive  |
| RoBERTa (LoRA)   | 78.00%   | Very High      | GPU-intensive  |
| 🤖 Benchmark DistilBERT | **90.00%** | Very High | GPU-intensive  |

---

## 📌 Key Insights

- Classical ML models perform surprisingly well, especially **Logistic Regression**.
- **DistilBERT** outperformed RoBERTa when fine-tuned with LoRA.
- **LoRA** significantly reduces the training cost for PLMs while preserving performance.
- The **benchmark model** (fine-tuned on SST-2) showed the best overall results but at a high resource cost.
- PLMs shine on **context-rich datasets**, while classical models remain efficient on smaller-scale tasks.

---

## ✅ Conclusion

- 🏆 **Best Classical Model**: Logistic Regression (89.27%)
- 🧠 **Best Fine-Tuned PLM**: DistilBERT + LoRA (87.15%)
- 👑 **Best Overall**: Hugging Face Benchmark Model (90.00%)
- ⚖️ Trade-off: PLMs offer better contextual understanding but demand more resources. Classical ML is efficient and interpretable for simpler tasks.

---

## 📚 References

- Vaswani et al., “Attention is All You Need”, 2017  
- Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer”, 2019  
- Hugging Face Model Hub: [`distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)

---


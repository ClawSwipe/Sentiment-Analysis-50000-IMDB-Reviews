# Sentiment Analysis 50000 IMDB Reviews
 ```markdown
# 🧠 NLP Text Classification with CNN, LSTM, and Hybrid Models

This project explores different deep learning architectures for text classification, focusing on **CNNs, LSTMs, and a hybrid LSTM+CNN model**. Using **GloVe embeddings** and **IMDB movie reviews**, we compare how each model performs in sentiment analysis.

## 📌 Features
- **CNN Model**: Captures local patterns in text.
- **LSTM Model**: Understands sequential dependencies.
- **LSTM + CNN Hybrid**: Combines both for better performance.
- **Pretrained GloVe Embeddings** for improved word representation.

## 🚀 Getting Started

### 1️⃣ Install Dependencies
```bash
pip install tensorflow numpy pandas scikit-learn
```

### 2️⃣ Prepare the Data
Download the GloVe embeddings:
- IMDB dataset: [`aclImdb`](https://ai.stanford.edu/~amaas/data/sentiment/)
- GloVe embeddings: [`glove.6B.100d.txt`](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)

Place them in the `aclImdb` folder.

### 3️⃣ Run the Model
```bash
python main.py
```
Choose from:
1️⃣ CNN  
2️⃣ LSTM  
3️⃣ LSTM + CNN  

## 📊 Results
Each model is trained and evaluated, printing:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report**

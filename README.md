# Sentiment_Analysis_Showdown_Classic_ML_VS_Modern_LLM
# Classic ML vs Gemini LLM: A Sentiment Analysis Showdown

## Project Overview
This project compares two approaches to **sentiment analysis** on text reviews:  

1. **Classic Machine Learning** – A supervised pipeline using **TF-IDF vectorization + Logistic Regression**.  
2. **Modern Large Language Model (LLM)** – A **zero-shot classification** approach using **Google Gemini** via API.  

The dataset is the [UCI Sentiment Labelled Sentences Dataset](https://archive.ics.uci.edu/ml/datasets/sentiment+labelled+sentences), containing **Amazon, Yelp, and IMDB reviews** (total 2,748 examples). Each review is labeled **positive (1)** or **negative (0)**.  

The goal is to **evaluate performance, speed, and trade-offs** between a traditional ML workflow and an LLM-based approach.


---

##  Workflow
1. **Data Preparation** – Loaded and merged Amazon, Yelp, IMDB reviews.  
2. **Feature Engineering** – Applied **TF-IDF vectorizer** (with stopword removal, min_df=2).  
3. **Classic ML Model** – Trained Logistic Regression with stratified train/test split.  
4. **Evaluation (Classic ML)** – Generated accuracy, precision, recall, F1 metrics.  
5. **Gemini Integration** – Used Google Generative AI (Gemini) API with prompt engineering.  
6. **Evaluation (Gemini)** – Converted outputs to labels, ran classification report.  
7. **Comparison** – Built a **side-by-side performance table** for both approaches.  

---

## Results
| Metric               | Logistic Regression 
|-----------------------|---------------------
| Accuracy              | 0.79
| Precision (Negative)  | 0.76             
| Recall (Negative)     | 0.85             
| F1 (Negative)         | 0.80            
| Precision (Positive)  | 0.83           
| Recall (Positive)     | 0.73             
| F1 (Positive)         | 0.78              

for gemini the results were 0.99 accurate.
---

## Reflection

### 1. Development Experience
- **Classic ML (TF-IDF + Logistic Regression):**  
  - Faster to set up once preprocessing pipeline was clear.  
  - Required careful handling of train/test split to avoid data leakage.  
  - Hyperparameters (e.g., `max_iter`) needed tuning for convergence.  
- **Gemini (LLM):**  
  - Very quick to integrate — no training required.  
  - Main challenge was **prompt engineering** (forcing the model to respond only with “Positive” or “Negative”).  
  - Added **error handling** for API timeouts and invalid responses.  

### 2. Performance
- **Logistic Regression** achieved **consistent accuracy** across Amazon, Yelp, IMDB.  
- **Gemini** was competitive but sometimes misclassified:  
  - Short sarcastic reviews.  
  - Reviews with mixed sentiment (e.g., “Good value, but terrible battery life”).  
- Gemini occasionally returned unexpected outputs (e.g., extra words), which required normalization.  

### 3. Control vs. Convenience
- **Classic ML:**  
  - Full control over preprocessing, model choice, hyperparameters.  
  - Transparent and explainable — important in regulated settings.  
  - Requires labeled data and training.  
- **Gemini:**  
  - Zero-shot → no need for labeled data.  
  - Convenient and flexible — can adapt to many tasks with prompt changes.  
  - Less interpretable and dependent on API cost/availability.  

### 4. Real-World Application
- **When to choose Classic ML:**  
  - Large-scale production where cost per prediction must be minimal.  
  - Environments requiring **explainability and reproducibility** (finance, healthcare).  
  - When a labeled dataset is already available.  
- **When to choose Gemini (LLM):**  
  - Rapid prototyping of NLP applications.  
  - Low-data scenarios where collecting training data is difficult.  
  - Complex tasks requiring **contextual understanding** beyond simple word patterns.  

---

## Key Learnings
- Merging all three datasets improved generalization vs single-source data.  
- LLMs like Gemini are powerful but require careful prompting and error handling.  
- Classic ML remains strong for simple binary classification — lower cost, faster inference.  
- The **showdown** highlighted the trade-off:  
  - Classic ML = **control, stability, low-cost**.  
  - Gemini = **convenience, adaptability, high compute/API dependency**.  

---

## Tech Stack
- Python 3.10  
- Pandas 2.2.2  
- Scikit-learn  
- Google Generative AI (Gemini API)  
- Google Colab  

---

##  Demo
- **Colab Notebook:** [View in Colab](https://colab.research.google.com/drive/12k-seQTMDgOCjd9adoXcSh5ZEwfkVrbJ?usp=sharing)  

---

## 👩‍💻 Author
**Fareena Shahbaz**  
[LinkedIn](www.linkedin.com/in/fareena-shahbaz-137264351) 

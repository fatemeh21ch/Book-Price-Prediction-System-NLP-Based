# 📚 Book Price Prediction using NLP & Machine Learning

This project aims to predict book prices based on metadata such as title, genre, author, and publication details using Natural Language Processing (NLP) techniques and machine learning models.

---

## 🚀 Project Overview

- **Goal:** Predict book prices more accurately using textual and categorical metadata.  
- **Data:** Excel datasets containing book information (titles, authors, genres, publication year, ratings, and reviews).  
- **Techniques Used:**
  - Data preprocessing (text cleaning, numerical extraction, normalization)
  - Feature engineering (CountVectorizer, FastText, BERT embeddings)
  - Machine learning models (Linear Regression, Fully Connected Neural Networks)
  - Evaluation using regression metrics

---

## 🔧 Preprocessing Steps

- Handling missing values (none found in dataset)  
- Extracting numerical values from mixed text/number fields  
- Visualizing correlations (e.g., Ratings–Price, Reviews–Price)  
- Encoding categorical features into numerical representations  
- Standardizing and normalizing data before model training  

---

## 🧠 Models Implemented

### Linear Regression
- Simple baseline with MSE as loss function  

### Fully Connected Neural Network
- Architecture: 3 hidden layers + Dropout  
- Optimizer: Adam (learning rate = 0.001)  
- Loss: Mean Squared Error (MSE)  
- Training/Test split: 80/20  
- Epochs: 100–200  

---

## 📊 Results

| Embedding Method | Model               | MSE     | MAE     | R² Score |
|-----------------|-------------------|---------|---------|----------|
| FastText        | Neural Network     | 0.4515  | 0.3474  | 0.4276   |
| FastText        | Linear Regression  | 0.5626  | 0.4378  | 0.2867   |
| CountVectorizer | Neural Network     | 0.4955  | 0.3467  | 0.3718   |
| CountVectorizer | Linear Regression  | 0.6821  | 0.5342  | 0.1353   |

**✅ Best Performing Model:** FastText + Neural Network (highest R², lowest errors)

---

## 📂 Project Structure

├── data/ # Raw and processed book datasets
├── notebooks/ # Jupyter notebooks with experiments
├── models/ # Trained models and evaluation
├── results/ # Plots and performance reports
└── README.md # Project documentation

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - NumPy, Pandas, Matplotlib (data handling & visualization)  
  - scikit-learn (CountVectorizer, regression models)  
  - PyTorch / TensorFlow (Neural Networks)  
  - FastText, BERT (embeddings)  

---

## 📈 Key Insights

- Textual embeddings significantly improve model accuracy compared to basic regression  
- FastText embeddings outperform CountVectorizer across most metrics  
- Neural networks achieve better performance than linear regression  

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork this repo and submit a pull request.

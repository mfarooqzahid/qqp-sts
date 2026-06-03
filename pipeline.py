import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords

# Download nltk resources if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class QQPPipeline:
    def __init__(self, data_path='data/questions.csv'):
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.tfidf = None
        self.log_reg = None
        self.sbert_model = None
        self.sbert_model_name = 'all-MiniLM-L6-v2'
        
    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove special chars except '?'
        text = re.sub(r"[^a-zA-Z0-9? ]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_and_split_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
            
        self.df = pd.read_csv(self.data_path)
        self.df['question1'] = self.df['question1'].apply(self.preprocess)
        self.df['question2'] = self.df['question2'].apply(self.preprocess)
        
        # 80/20 stratified split
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=0.2, random_state=42, stratify=self.df['is_duplicate']
        )
        return self.train_df, self.test_df

    def train_tfidf_models(self):
        # Combine all questions from train set for TF-IDF fitting
        all_questions = pd.concat([self.train_df['question1'], self.train_df['question2']])
        
        self.tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
        self.tfidf.fit(all_questions)
        
        # Features for Logistic Regression
        q1_tfidf = self.tfidf.transform(self.train_df['question1'])
        q2_tfidf = self.tfidf.transform(self.train_df['question2'])
        
        # Combine features: absolute difference and element-wise product are common for STS
        # Here we follow a simpler approach: concatenate or just difference
        # But usually, it's |q1-q2| and q1*q2
        from scipy.sparse import hstack
        
        diff = abs(q1_tfidf - q2_tfidf)
        prod = q1_tfidf.multiply(q2_tfidf)
        X_train = hstack([diff, prod])
        
        self.log_reg = LogisticRegression(C=1.0, max_iter=1000)
        self.log_reg.fit(X_train, self.train_df['is_duplicate'])

    def load_sbert(self):
        self.sbert_model = SentenceTransformer(self.sbert_model_name)

    def model1_predict(self, q1, q2):
        # TF-IDF Cosine Similarity baseline
        q1_processed = self.preprocess(q1)
        q2_processed = self.preprocess(q2)
        
        vecs = self.tfidf.transform([q1_processed, q2_processed])
        similarity = cosine_similarity(vecs[0], vecs[1])[0][0]
        label = 1 if similarity >= 0.5 else 0
        return similarity, label

    def model2_predict(self, q1, q2):
        # TF-IDF + Logistic Regression
        q1_processed = self.preprocess(q1)
        q2_processed = self.preprocess(q2)
        
        q1_vec = self.tfidf.transform([q1_processed])
        q2_vec = self.tfidf.transform([q2_processed])
        
        from scipy.sparse import hstack
        diff = abs(q1_vec - q2_vec)
        prod = q1_vec.multiply(q2_vec)
        X = hstack([diff, prod])
        
        prob = self.log_reg.predict_proba(X)[0][1]
        label = self.log_reg.predict(X)[0]
        return prob, label

    def model3_predict(self, q1, q2):
        # Sentence-BERT Cosine Similarity
        if self.sbert_model is None:
            self.load_sbert()
            
        embeddings = self.sbert_model.encode([q1, q2], batch_size=64)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        label = 1 if similarity >= 0.5 else 0
        return similarity, label

    def predict(self, q1, q2, model_type='model3'):
        """
        Returns similarity score, binary label, and model used.
        """
        if model_type == 'model1':
            score, label = self.model1_predict(q1, q2)
            name = "TF-IDF Cosine Similarity (Baseline)"
        elif model_type == 'model2':
            score, label = self.model2_predict(q1, q2)
            name = "TF-IDF + Logistic Regression"
        else:
            score, label = self.model3_predict(q1, q2)
            name = "Sentence-BERT (MiniLM)"
            
        return {
            'score': float(score),
            'label': int(label),
            'model': name,
            'verdict': 'Duplicate' if label == 1 else 'Not Duplicate'
        }

if __name__ == "__main__":
    # For testing if file exists
    pipeline = QQPPipeline()
    try:
        pipeline.load_and_split_data()
        pipeline.train_tfidf_models()
        print("Pipeline initialized and models trained.")
    except Exception as e:
        print(f"Initialization failed: {e}")

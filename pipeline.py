import pandas as pd
import numpy as np
import re
import os
import joblib
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
        
        # New model components
        self.model1_threshold = 0.28  # Calibrated on train split
        self.model3_threshold = 0.73  # Calibrated on train split
        self.sbert_classifier = None
        self.ensemble_sbert_weight = 0.6
        
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
        # Keep this for backward compatibility (called by original scripts)
        # Combine all questions from train set for TF-IDF fitting
        all_questions = pd.concat([self.train_df['question1'], self.train_df['question2']])
        
        self.tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
        self.tfidf.fit(all_questions)
        
        # Features for Logistic Regression
        q1_tfidf = self.tfidf.transform(self.train_df['question1'])
        q2_tfidf = self.tfidf.transform(self.train_df['question2'])
        
        from scipy.sparse import hstack
        diff = abs(q1_tfidf - q2_tfidf)
        prod = q1_tfidf.multiply(q2_tfidf)
        X_train = hstack([diff, prod])
        
        self.log_reg = LogisticRegression(C=1.0, max_iter=1000)
        self.log_reg.fit(X_train, self.train_df['is_duplicate'])

        # Calibrate threshold for Model 1 (TF-IDF Cosine Baseline)
        train_scores_m1 = np.array(q1_tfidf.multiply(q2_tfidf).sum(axis=1)).flatten()
        y_train = self.train_df['is_duplicate'].values
        
        best_t_m1 = 0.5
        best_f1_m1 = 0.0
        from sklearn.metrics import f1_score
        for t in np.linspace(0.1, 0.9, 81):
            preds = (train_scores_m1 >= t).astype(int)
            f1 = f1_score(y_train, preds, zero_division=0)
            if f1 > best_f1_m1:
                best_f1_m1 = f1
                best_t_m1 = t
        self.model1_threshold = float(best_t_m1)

    def train_models(self):
        # Trains all models including SBERT models
        print("Training TF-IDF Models...")
        self.train_tfidf_models()
        
        print("Loading Sentence-BERT...")
        if self.sbert_model is None:
            self.load_sbert()
            
        print("Encoding training questions with SBERT (this may take a minute)...")
        train_q1_emb = self.sbert_model.encode(self.train_df['question1'].tolist(), batch_size=64)
        train_q2_emb = self.sbert_model.encode(self.train_df['question2'].tolist(), batch_size=64)
        
        from sklearn.metrics.pairwise import paired_cosine_distances
        train_scores_m3 = 1 - paired_cosine_distances(train_q1_emb, train_q2_emb)
        y_train = self.train_df['is_duplicate'].values
        
        # Calibrate Model 3 (SBERT Cosine) threshold
        print("Calibrating SBERT threshold...")
        best_t_m3 = 0.5
        best_f1_m3 = 0.0
        from sklearn.metrics import f1_score
        for t in np.linspace(0.1, 0.9, 81):
            preds = (train_scores_m3 >= t).astype(int)
            f1 = f1_score(y_train, preds, zero_division=0)
            if f1 > best_f1_m3:
                best_f1_m3 = f1
                best_t_m3 = t
        self.model3_threshold = float(best_t_m3)
        
        # Train Model 4 (SBERT Supervised Classifier)
        print("Training SBERT Supervised Classifier...")
        diff_sbert = np.abs(train_q1_emb - train_q2_emb)
        prod_sbert = train_q1_emb * train_q2_emb
        cos_sbert = train_scores_m3.reshape(-1, 1)
        X_train_m4 = np.hstack([diff_sbert, prod_sbert, cos_sbert])
        
        self.sbert_classifier = LogisticRegression(C=1.0, max_iter=1000)
        self.sbert_classifier.fit(X_train_m4, y_train)

    def save_models(self, directory='outputs'):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        joblib.dump(self.tfidf, os.path.join(directory, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.log_reg, os.path.join(directory, 'model2_logreg.joblib'))
        
        if self.sbert_classifier is not None:
            joblib.dump(self.sbert_classifier, os.path.join(directory, 'model4_sbert_logreg.joblib'))
            
        thresholds = {
            'model1_threshold': self.model1_threshold,
            'model3_threshold': self.model3_threshold
        }
        joblib.dump(thresholds, os.path.join(directory, 'thresholds.joblib'))
        print(f"Models and thresholds saved to {directory}/")

    def load_models(self, directory='outputs'):
        try:
            tfidf_path = os.path.join(directory, 'tfidf_vectorizer.joblib')
            m2_path = os.path.join(directory, 'model2_logreg.joblib')
            m4_path = os.path.join(directory, 'model4_sbert_logreg.joblib')
            thresholds_path = os.path.join(directory, 'thresholds.joblib')
            
            if not (os.path.exists(tfidf_path) and os.path.exists(m2_path) and os.path.exists(thresholds_path)):
                return False
                
            self.tfidf = joblib.load(tfidf_path)
            self.log_reg = joblib.load(m2_path)
            
            thresholds = joblib.load(thresholds_path)
            self.model1_threshold = thresholds.get('model1_threshold', 0.28)
            self.model3_threshold = thresholds.get('model3_threshold', 0.73)
            
            if os.path.exists(m4_path):
                self.sbert_classifier = joblib.load(m4_path)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def load_sbert(self):
        self.sbert_model = SentenceTransformer(self.sbert_model_name)

    def model1_predict(self, q1, q2):
        # TF-IDF Cosine Similarity baseline with optimized threshold
        q1_processed = self.preprocess(q1)
        q2_processed = self.preprocess(q2)
        
        vecs = self.tfidf.transform([q1_processed, q2_processed])
        similarity = cosine_similarity(vecs[0], vecs[1])[0][0]
        label = 1 if similarity >= self.model1_threshold else 0
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
        # Sentence-BERT Cosine Similarity with calibrated threshold
        if self.sbert_model is None:
            self.load_sbert()
            
        embeddings = self.sbert_model.encode([q1, q2], batch_size=64)
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        label = 1 if similarity >= self.model3_threshold else 0
        return similarity, label

    def model4_predict(self, q1, q2):
        # Sentence-BERT Supervised Classifier
        if self.sbert_model is None:
            self.load_sbert()
        if self.sbert_classifier is None:
            # Fallback if classifier is not trained/loaded
            score, label = self.model3_predict(q1, q2)
            return score, label
            
        embeddings = self.sbert_model.encode([q1, q2], batch_size=64)
        emb1, emb2 = embeddings[0], embeddings[1]
        
        diff = np.abs(emb1 - emb2)
        prod = emb1 * emb2
        cos_sim = cosine_similarity([emb1], [emb2])[0][0]
        
        X = np.hstack([diff, prod, cos_sim]).reshape(1, -1)
        prob = self.sbert_classifier.predict_proba(X)[0][1]
        label = self.sbert_classifier.predict(X)[0]
        return prob, label

    def model5_predict(self, q1, q2):
        # Ensemble (Model 2 + Model 4)
        prob2, _ = self.model2_predict(q1, q2)
        prob4, _ = self.model4_predict(q1, q2)
        
        prob = (self.ensemble_sbert_weight * prob4) + ((1 - self.ensemble_sbert_weight) * prob2)
        label = 1 if prob >= 0.5 else 0
        return prob, label

    def predict(self, q1, q2, model_type='model5'):
        """
        Returns similarity score, binary label, and model used.
        """
        if model_type == 'model1':
            score, label = self.model1_predict(q1, q2)
            name = "TF-IDF Cosine (Calibrated)"
            threshold = self.model1_threshold
        elif model_type == 'model2':
            score, label = self.model2_predict(q1, q2)
            name = "TF-IDF + Logistic Regression"
            threshold = 0.5
        elif model_type == 'model3':
            score, label = self.model3_predict(q1, q2)
            name = "Sentence-BERT Cosine (Calibrated)"
            threshold = self.model3_threshold
        elif model_type == 'model4':
            score, label = self.model4_predict(q1, q2)
            name = "Sentence-BERT Supervised Classifier"
            threshold = 0.5
        else:
            score, label = self.model5_predict(q1, q2)
            name = "Lexical + Semantic Ensemble"
            threshold = 0.5
            
        return {
            'score': float(score),
            'label': int(label),
            'model': name,
            'threshold': float(threshold),
            'verdict': 'Duplicate' if label == 1 else 'Not Duplicate'
        }

if __name__ == "__main__":
    pipeline = QQPPipeline()
    try:
        pipeline.load_and_split_data()
        pipeline.train_models()
        pipeline.save_models()
        print("Pipeline initialized, all models trained and saved successfully.")
    except Exception as e:
        print(f"Initialization failed: {e}")

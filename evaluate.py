import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from pipeline import QQPPipeline
import os

def evaluate():
    pipeline = QQPPipeline()
    print("Loading data and training models...")
    pipeline.load_and_split_data()
    pipeline.train_tfidf_models()
    pipeline.load_sbert()
    
    test_df = pipeline.test_df
    y_true = test_df['is_duplicate'].values
    
    models = ['model1', 'model2', 'model3']
    results = {}
    
    plt.figure(figsize=(10, 8))
    
    for m in models:
        print(f"Evaluating {m}...")
        y_scores = []
        y_pred = []
        
        # SBERT might be slow if we do it one by one, but pipeline has a batch size constraint of 64
        # However, our predict() is one-by-one. 
        # For evaluation, let's optimize Model 3 (SBERT) to use batches if possible, 
        # or just run it as is since it's only ~2.4k rows.
        
        if m == 'model3':
            # Batch encode for SBERT to respect the batch_size=64 constraint
            q1_list = test_df['question1'].tolist()
            q2_list = test_df['question2'].tolist()
            
            emb1 = pipeline.sbert_model.encode(q1_list, batch_size=64, show_progress_bar=True)
            emb2 = pipeline.sbert_model.encode(q2_list, batch_size=64, show_progress_bar=True)
            
            from sklearn.metrics.pairwise import paired_cosine_distances
            # paired_cosine_distances returns 1 - cosine_similarity
            y_scores = 1 - paired_cosine_distances(emb1, emb2)
            y_pred = (y_scores >= 0.5).astype(int)
        elif m == 'model1':
            # Optimized TF-IDF cosine similarity for entire test set
            q1_tfidf = pipeline.tfidf.transform(test_df['question1'])
            q2_tfidf = pipeline.tfidf.transform(test_df['question2'])
            
            # Row-wise cosine similarity
            # cos_sim(A, B) = dot(A, B) / (||A|| * ||B||)
            # TfidfVectorizer outputs L2 normalized vectors, so dot product is cosine similarity
            y_scores = np.array(q1_tfidf.multiply(q2_tfidf).sum(axis=1)).flatten()
            y_pred = (y_scores >= 0.5).astype(int)
        elif m == 'model2':
            # TF-IDF + LogReg
            q1_tfidf = pipeline.tfidf.transform(test_df['question1'])
            q2_tfidf = pipeline.tfidf.transform(test_df['question2'])
            
            from scipy.sparse import hstack
            diff = abs(q1_tfidf - q2_tfidf)
            prod = q1_tfidf.multiply(q2_tfidf)
            X_test = hstack([diff, prod])
            
            y_scores = pipeline.log_reg.predict_proba(X_test)[:, 1]
            y_pred = pipeline.log_reg.predict(X_test)
            
        # Calculate Metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_scores)
        }
        results[m] = metrics
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f"{m} (AUC = {metrics['ROC-AUC']:.2f})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {m}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'outputs/cm_{m}.png')
        plt.close()

    # Save ROC Plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend()
    plt.savefig('outputs/roc_curves.png')
    plt.close()

    # Output Metrics to CSV/Table
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv('outputs/metrics.csv')
    print("\n--- Evaluation Results ---")
    print(metrics_df)
    print("\nPlots and metrics saved to outputs/")

if __name__ == "__main__":
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    try:
        evaluate()
    except Exception as e:
        print(f"Evaluation failed: {e}")

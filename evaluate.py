import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from pipeline import QQPPipeline
import os
import argparse

def evaluate(force_train=False):
    pipeline = QQPPipeline()
    print("Loading data...")
    pipeline.load_and_split_data()
    
    # Check if pre-trained models exist, otherwise train
    models_loaded = False
    if not force_train:
        print("Checking for pre-trained models...")
        models_loaded = pipeline.load_models()
        
    if not models_loaded:
        print("Training all models from scratch...")
        pipeline.train_models()
        pipeline.save_models()
    else:
        print("Loaded pre-trained models successfully.")
        
    if pipeline.sbert_model is None:
        print("Loading Sentence-BERT for evaluation...")
        pipeline.load_sbert()
        
    test_df = pipeline.test_df
    y_true = test_df['is_duplicate'].values
    
    models = ['model1', 'model2', 'model3', 'model4', 'model5']
    results = {}
    
    # We will compute SBERT embeddings once to reuse them across Model 3, 4, 5
    print("Computing SBERT embeddings for test set...")
    q1_list = test_df['question1'].tolist()
    q2_list = test_df['question2'].tolist()
    emb1 = pipeline.sbert_model.encode(q1_list, batch_size=64, show_progress_bar=True)
    emb2 = pipeline.sbert_model.encode(q2_list, batch_size=64, show_progress_bar=True)
    
    from sklearn.metrics.pairwise import paired_cosine_distances
    sbert_test_cos = 1 - paired_cosine_distances(emb1, emb2)
    
    # Store probability/similarity scores to use for ensemble and plotting
    saved_scores = {}
    
    plt.figure(figsize=(10, 8))
    
    for m in models:
        print(f"Evaluating {m}...")
        y_scores = []
        y_pred = []
        
        if m == 'model1':
            # Calibrated TF-IDF cosine similarity
            q1_tfidf = pipeline.tfidf.transform(test_df['question1'])
            q2_tfidf = pipeline.tfidf.transform(test_df['question2'])
            y_scores = np.array(q1_tfidf.multiply(q2_tfidf).sum(axis=1)).flatten()
            y_pred = (y_scores >= pipeline.model1_threshold).astype(int)
            
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
            saved_scores['model2'] = y_scores
            
        elif m == 'model3':
            # Calibrated SBERT Cosine Similarity
            y_scores = sbert_test_cos
            y_pred = (y_scores >= pipeline.model3_threshold).astype(int)
            
        elif m == 'model4':
            # SBERT Supervised
            diff_sbert = np.abs(emb1 - emb2)
            prod_sbert = emb1 * emb2
            cos_sbert = sbert_test_cos.reshape(-1, 1)
            X_test_m4 = np.hstack([diff_sbert, prod_sbert, cos_sbert])
            
            y_scores = pipeline.sbert_classifier.predict_proba(X_test_m4)[:, 1]
            y_pred = pipeline.sbert_classifier.predict(X_test_m4)
            saved_scores['model4'] = y_scores
            
        elif m == 'model5':
            # Ensemble (Model 2 + Model 4)
            scores2 = saved_scores['model2']
            scores4 = saved_scores['model4']
            y_scores = (pipeline.ensemble_sbert_weight * scores4) + ((1 - pipeline.ensemble_sbert_weight) * scores2)
            y_pred = (y_scores >= 0.5).astype(int)
            
        # Calculate Metrics
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1': f1_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_scores)
        }
        results[m] = metrics
        
        # ROC Curve plotting
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        model_display_names = {
            'model1': 'Model 1: TF-IDF Cosine (Calibrated)',
            'model2': 'Model 2: TF-IDF + LogReg',
            'model3': 'Model 3: SBERT Cosine (Calibrated)',
            'model4': 'Model 4: SBERT Supervised',
            'model5': 'Model 5: Lexical+Semantic Ensemble'
        }
        plt.plot(fpr, tpr, label=f"{model_display_names[m]} (AUC = {metrics['ROC-AUC']:.3f})")
        
        # Confusion Matrix plotting
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_display_names[m]}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Ensure target directories exist
        os.makedirs('public/outputs', exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        plt.savefig(f'public/outputs/cm_{m}.png', bbox_inches='tight')
        plt.savefig(f'outputs/cm_{m}.png', bbox_inches='tight')
        plt.close()
        
    # Save ROC Plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models')
    plt.legend(loc='lower right')
    plt.savefig('public/outputs/roc_curves.png', bbox_inches='tight', dpi=150)
    plt.savefig('outputs/roc_curves.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Save Metrics CSV
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = 'ModelKey'
    metrics_df.to_csv('public/outputs/metrics.csv')
    metrics_df.to_csv('outputs/metrics.csv')
    
    print("\n--- Evaluation Results ---")
    print(metrics_df)
    print("\nPlots and metrics saved to public/outputs/ and outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-train', action='store_true', help='Force retraining of all models')
    args = parser.parse_args()
    
    try:
        evaluate(force_train=args.force_train)
    except Exception as e:
        print(f"Evaluation failed: {e}")

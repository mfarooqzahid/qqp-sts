from datasets import load_dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def download_and_sample():
    print("Downloading QQP dataset from Hugging Face...")
    # Load the GLUE QQP dataset
    dataset = load_dataset('glue', 'qqp')
    
    # Convert to pandas
    df = pd.DataFrame(dataset['train'])
    
    # Rename 'label' to 'is_duplicate' for consistency with project requirements
    df = df.rename(columns={'label': 'is_duplicate'})
    
    # Drop rows with missing questions
    df = df.dropna(subset=['question1', 'question2'])
    
    print(f"Total entries downloaded: {len(df)}")
    
    # Take a 12k stratified sample
    # Using train_test_split as a sampling tool to ensure stratification
    _, sample_df = train_test_split(
        df, 
        test_size=12000 / len(df), 
        stratify=df['is_duplicate'], 
        random_state=42
    )
    
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Save to CSV
    output_path = 'data/questions.csv'
    sample_df[['question1', 'question2', 'is_duplicate']].to_csv(output_path, index=False)
    
    print(f"Sampled {len(sample_df)} entries and saved to {output_path}")

if __name__ == "__main__":
    download_and_sample()

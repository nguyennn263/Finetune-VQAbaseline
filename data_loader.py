#!/usr/bin/env python3
"""
Data loader for Vietnamese VQA dataset
Customize this file to load your specific VQA dataset
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any

def load_vqa_data(data_path: str, data_format: str = 'auto') -> pd.DataFrame:
    """
    Load VQA data from various formats
    
    Args:
        data_path: Path to your data file
        data_format: Format of data ('csv', 'json', 'auto')
    
    Returns:
        DataFrame with columns: image_name, question, answers
    """
    
    if data_format == 'auto':
        if data_path.endswith('.csv'):
            data_format = 'csv'
        elif data_path.endswith('.json'):
            data_format = 'json'
        else:
            raise ValueError("Cannot auto-detect format. Please specify 'csv' or 'json'")
    
    if data_format == 'csv':
        df = pd.read_csv(data_path)
    elif data_format == 'json':
        df = pd.read_json(data_path)
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'json'")
    
    return df

def process_multiple_answers(answers_text: str) -> List[str]:
    """
    Process multiple answers from various text formats
    
    Expected input formats:
    - "answer1|answer2|answer3" (pipe-separated)
    - "['answer1', 'answer2', 'answer3']" (string representation of list)
    - "answer1, answer2, answer3" (comma-separated)
    """
    
    if pd.isna(answers_text) or answers_text == '':
        return [""]
    
    # Convert to string if not already
    answers_text = str(answers_text).strip()
    
    # Handle list-like strings
    if answers_text.startswith('[') and answers_text.endswith(']'):
        try:
            import ast
            return ast.literal_eval(answers_text)
        except:
            # Remove brackets and split by comma
            answers_text = answers_text[1:-1]
            return [ans.strip().strip('"\'') for ans in answers_text.split(',')]
    
    # Handle pipe-separated
    elif '|' in answers_text:
        return [ans.strip() for ans in answers_text.split('|') if ans.strip()]
    
    # Handle comma-separated
    elif ',' in answers_text:
        return [ans.strip() for ans in answers_text.split(',') if ans.strip()]
    
    # Single answer
    else:
        return [answers_text]

def standardize_vqa_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame to expected VQA format
    
    Expected columns after standardization:
    - image_name: filename of the image
    - question: the question text  
    - answers: list of multiple correct answers
    - ground_truth: primary answer for training
    """
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Standardize column names (handle common variations)
    column_mapping = {
        'image': 'image_name',
        'img': 'image_name', 
        'image_file': 'image_name',
        'filename': 'image_name',
        
        'query': 'question',
        'text': 'question',
        'question_text': 'question',
        
        'answer': 'answers',
        'response': 'answers',
        'target': 'answers',
        'label': 'answers',
        'correct_answers': 'answers',
        'all_answers': 'answers'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    
    # Ensure required columns exist
    required_columns = ['image_name', 'question', 'answers']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Process answers to ensure they're lists
    if 'answers' in df.columns:
        df['answers'] = df['answers'].apply(process_multiple_answers)
    
    # Create ground_truth as primary answer if not exists
    if 'ground_truth' not in df.columns:
        df['ground_truth'] = df['answers'].apply(lambda x: x[0] if x else "")
    
    # Ensure image_name has proper extension
    if not df['image_name'].iloc[0].endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        print("Warning: Image names might be missing file extensions")
    
    return df

def create_sample_data():
    """Create sample VQA data for testing"""
    
    sample_data = {
        'image_name': [
            'sample1.jpg', 'sample2.jpg', 'sample3.jpg', 
            'sample4.jpg', 'sample5.jpg'
        ],
        'question': [
            'Trong ảnh có gì?',
            'Màu sắc chủ đạo là gì?', 
            'Có bao nhiêu người trong ảnh?',
            'Đây là hoạt động gì?',
            'Thời tiết như thế nào?'
        ],
        'answers': [
            'một chiếc xe|xe hơi|ô tô|xe cộ',
            'màu xanh|xanh lá cây|xanh',
            'hai người|2 người|hai|2',
            'đi bộ|tản bộ|đi dạo|dạo chơi',
            'nắng|trời nắng|thời tiết đẹp|tốt'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    return standardize_vqa_dataframe(df)

def validate_vqa_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate and analyze VQA dataset"""
    
    validation_results = {
        'total_samples': len(df),
        'issues': [],
        'statistics': {}
    }
    
    # Check for missing values
    for col in ['image_name', 'question', 'answers']:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            validation_results['issues'].append(f"Missing values in {col}: {missing_count}")
    
    # Check answer counts
    answer_counts = df['answers'].apply(len)
    validation_results['statistics']['avg_answers_per_question'] = answer_counts.mean()
    validation_results['statistics']['min_answers'] = answer_counts.min()
    validation_results['statistics']['max_answers'] = answer_counts.max()
    
    # Check for empty questions/answers
    empty_questions = df['question'].str.strip().eq('').sum()
    if empty_questions > 0:
        validation_results['issues'].append(f"Empty questions: {empty_questions}")
    
    # Check unique images
    unique_images = df['image_name'].nunique()
    validation_results['statistics']['unique_images'] = unique_images
    validation_results['statistics']['questions_per_image'] = len(df) / unique_images
    
    return validation_results

# Example usage functions
def load_your_data(data_path: str) -> pd.DataFrame:
    """
    Template function to load your specific VQA data
    Customize this function based on your data format
    """
    
    # Example for CSV format:
    # df = pd.read_csv(data_path)
    
    # Example for JSON format:
    # df = pd.read_json(data_path)
    
    # Example for custom JSON format:
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    # df = pd.DataFrame(data)
    
    # For now, return sample data
    print("Using sample data. Please modify load_your_data() function for your actual data.")
    return create_sample_data()

if __name__ == "__main__":
    # Example usage
    print("VQA Data Loader Example")
    print("=" * 40)
    
    # Create sample data
    df = create_sample_data()
    
    print(f"Loaded {len(df)} samples")
    print("\nSample data:")
    print(df.head())
    
    print("\nValidation results:")
    results = validate_vqa_data(df)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print("\nTo use with your data:")
    print("1. Modify load_your_data() function")
    print("2. Ensure your data has columns: image_name, question, answers")
    print("3. Answers should be multiple correct responses separated by '|' or as a list")

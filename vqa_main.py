#!/usr/bin/env python3
"""
Mai    
    # Load your actual VQA dataset
    print("Loading VQA dataset...")
    
    # Replace with your actual data file path
    try:
        # Try to load the preprocessed CSV file first
        df = pd.read_csv('/home/nguyennn263/Documents/Thesis/Fintune/Dataset/text/text/evaluate_60k_data_balanced_preprocessed.csv')
        print(f"Loaded CSV data: {len(df)} samples")
    except FileNotFoundError:
        print("CSV file not found, trying JSON...")
        try:
            # If CSV not found, try JSON
            df = pd.read_json('/home/nguyennn263/Documents/Thesis/Fintune/Dataset/text/text/raw_qa_5.json')
            print(f"Loaded JSON data: {len(df)} samples")
        except:
            print("No data files found, using mock data...")
            # Fallback to mock data
            mock_data = {
                'image_name': ['image1.jpg', 'image2.jpg', 'image3.jpg'] * 100,
                'question': ['Trong ảnh có gì?', 'Màu gì chiếm ưu thế?', 'Có bao nhiêu người?'] * 100,
                'answers': [['một chiếc xe', 'xe hơi', 'ô tô'], ['màu xanh', 'xanh lá'], ['hai người', '2']] * 100
            }
            df = pd.DataFrame(mock_data)
    
    # Convert to questions format
    questions = prepare_data_from_dataframe(df)ese VQA training using CLIP, XLM-RoBERTa, and mT5
"""
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, CLIPProcessor
import warnings
warnings.filterwarnings('ignore')

from cxmt5.config import get_improved_config
from cxmt5.cxmt5 import (
    VietnameseVQADataset, 
    VietnameseVQAModel, 
    VQATrainer,
    prepare_data_from_dataframe
)

def load_and_prepare_data(config):
    """Load and prepare VQA dataset"""
    
    # You'll need to replace this with your actual data loading
    # Example data structure expected:
    # - image_name: filename of the image
    # - question: the question text
    # - answers: list of possible answers (for evaluation)
    # - ground_truth: the primary answer (for training)
    
    # Mock data for demonstration - replace with your actual data loading
    print("Loading VQA dataset...")
    
    # Replace this section with your actual data loading code
    # For example:
    df = pd.read_csv('/kaggle/input/auto-vivqa/text/text/evaluate_60k_data_balanced_preprocessed.csv')
    # or 
    # df = pd.read_json('your_vqa_dataset.json')
    # data = pd.read_csv('/home/nguyennn263/Documents/Thesis/Fintune/Dataset/text/text/raw_qa_5.json')
    # Mock data structure - replace with your actual data
    # mock_data = {
    #     'image_name': ['image1.jpg', 'image2.jpg', 'image3.jpg'] * 100,
    #     'question': ['Trong ảnh có gì?', 'Màu gì chiếm ưu thế?', 'Có bao nhiêu người?'] * 100,
    #     'answers': [['một chiếc xe', 'xe hơi', 'ô tô'], ['màu xanh', 'xanh lá'], ['hai người', '2']] * 100
    # }
    
    # df = pd.DataFrame(mock_data)
    
    # Convert to questions format
    questions = prepare_data_from_dataframe(df)
    
    print(f"Loaded {len(questions)} questions")
    
    return questions

def setup_models_and_tokenizers(config):
    """Setup model components"""
    
    print("Loading tokenizers and processors...")
    
    # Question tokenizer (XLM-RoBERTa)
    question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
    
    # Answer tokenizer (mT5)
    answer_tokenizer = AutoTokenizer.from_pretrained(config['decoder_model'])
    
    # CLIP processor for images
    clip_processor = CLIPProcessor.from_pretrained(config['vision_model'])
    
    print("Initializing VQA model...")
    model = VietnameseVQAModel(config)
    
    return model, question_tokenizer, answer_tokenizer, clip_processor

def create_dataloaders(questions, config, question_tokenizer, answer_tokenizer, clip_processor):
    """Create train and validation dataloaders"""
    
    # Create dataset
    full_dataset = VietnameseVQADataset(
        questions=questions,
        image_dir=config['image_dir'],
        question_tokenizer=question_tokenizer,
        answer_tokenizer=answer_tokenizer,
        clip_processor=clip_processor,
        max_length=config['max_length']
    )
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Created dataloaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader

def main():
    """Main training function"""
    
    print("="*80)
    print("Vietnamese VQA Training with CLIP + XLM-RoBERTa + mT5")
    print("="*80)
    
    # Load configuration
    config = get_improved_config()
    
    print(f"Configuration:")
    print(f"  Vision Model: {config['vision_model']}")
    print(f"  Text Model: {config['text_model']}")
    print(f"  Decoder Model: {config['decoder_model']}")
    print(f"  Device: {config['device']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    
    # Check device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    questions = load_and_prepare_data(config)
    
    # Setup models
    model, question_tokenizer, answer_tokenizer, clip_processor = setup_models_and_tokenizers(config)
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        questions, config, question_tokenizer, answer_tokenizer, clip_processor
    )
    
    # Initialize trainer
    trainer = VQATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    
    # Start training
    print(f"\nStarting training for {config['num_epochs']} epochs...")
    print("-" * 80)
    
    try:
        best_score = trainer.train(config['num_epochs'])
        
        print(f"\n{'='*80}")
        print(f"Training completed successfully!")
        print(f"Best VQA Score: {best_score:.4f}")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

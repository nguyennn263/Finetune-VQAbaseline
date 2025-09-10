"""
Enhanced Vietnamese VQA Training Script with Resume Capability

USAGE FOR RESUME TRAINING:
1. Set RESUME_TRAINING = True in main() function
2. Set CHECKPOINT_PATH to your checkpoint file (e.g., "checkpoints/best_fuzzy_model.pth")  
3. Set START_EPOCH to the next epoch number (e.g., 3 to continue after epoch 2)
4. Optionally adjust TOTAL_EPOCHS to train for more epochs
5. Run: python main.py

The script will automatically:
- Load the saved model weights
- Resume from the specified epoch
- Continue training with the remaining epochs
- Show training improvement since checkpoint
"""

from cxmt5.config import get_improved_config
from cxmt5.model import ImprovedVietnameseVQAModel, normalize_vietnamese_answer
from cxmt5.cxmt5 import VietnameseVQADataset, VietnameseVQAModel, VQATrainer, prepare_data_from_dataframe
from transformers import (
    CLIPProcessor, CLIPModel,
    XLMRobertaTokenizer, XLMRobertaModel,
    T5ForConditionalGeneration, T5Tokenizer,
    AutoTokenizer, AutoModel
)
import pandas as pd
from torch.utils.data import DataLoader
import torch
import warnings
warnings.filterwarnings('ignore')


print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())



def analyze_data_balance(questions):
    """Analyze answer distribution for balance with multiple answers support"""
    from collections import Counter
    from cxmt5.model import normalize_vietnamese_answer
    
    # Collect all answers (including all 5 per question)
    all_answers = []
    for q in questions:
        if 'all_correct_answers' in q and q['all_correct_answers']:
            # Add all 5 correct answers
            all_answers.extend([normalize_vietnamese_answer(ans) for ans in q['all_correct_answers']])
        else:
            # Fallback to ground_truth
            all_answers.append(normalize_vietnamese_answer(q['ground_truth']))
    
    answer_counts = Counter(all_answers)
    
    print(f"\nData Balance Analysis (Multiple Answers):")
    print(f"  Total questions: {len(questions):,}")
    print(f"  Total answer instances: {len(all_answers):,}")
    print(f"  Average answers per question: {len(all_answers) / len(questions):.2f}")
    print(f"  Unique answers: {len(answer_counts):,}")
    print(f"  Top 10 most common answers:")
    
    for answer, count in answer_counts.most_common(10):
        percentage = (count / len(all_answers)) * 100
        print(f"    '{answer}': {count} ({percentage:.2f}%)")
    
    # Check for severe imbalance
    most_common_count = answer_counts.most_common(1)[0][1]
    imbalance_ratio = most_common_count / len(all_answers)
    
    if imbalance_ratio > 0.2:  # Lower threshold for multiple answers
        print(f"Severe imbalance detected: {imbalance_ratio:.2f} of answers are the same")
    else:
        print(f"Data balance looks good: {imbalance_ratio:.2f}")

    return answer_counts


def list_available_checkpoints():
    """List all available checkpoint files"""
    import os
    import glob
    
    checkpoint_patterns = [
        "checkpoints/*.pth",
        "checkpoints/*.pt", 
        "*.pth",
        "*.pt",
        "*checkpoint*",
        "*epoch*"
    ]
    
    print(f"\nScanning for available checkpoints...")
    all_checkpoints = []
    
    for pattern in checkpoint_patterns:
        checkpoints = glob.glob(pattern)
        all_checkpoints.extend(checkpoints)
    
    # Remove duplicates and sort
    all_checkpoints = sorted(list(set(all_checkpoints)))
    
    if all_checkpoints:
        print(f"Found {len(all_checkpoints)} checkpoint file(s):")
        for i, checkpoint in enumerate(all_checkpoints, 1):
            size = os.path.getsize(checkpoint) / (1024*1024)  # MB
            mtime = os.path.getmtime(checkpoint)
            import datetime
            mod_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {i}. {checkpoint}")
            print(f"     Size: {size:.1f} MB, Modified: {mod_time}")
        
        print(f"\nRecommended checkpoint: {all_checkpoints[0]}")
        return all_checkpoints
    else:
        print(f"No checkpoint files found.")
        return []


def main():
    """Enhanced main training function with multiple answers support"""
    
    # Load improved configuration
    config = get_improved_config()
    
    # Configuration for resuming training
    # ================================
    # Để tiếp tục training từ checkpoint:
    # 1. Set RESUME_TRAINING = True
    # 2. Điều chỉnh CHECKPOINT_PATH tới file checkpoint của bạn  
    # 3. Set START_EPOCH = epoch muốn bắt đầu (ví dụ: 3 để tiếp tục sau epoch 2)
    # 4. Có thể điều chỉnh TOTAL_EPOCHS để training thêm nhiều epoch hơn
    
    RESUME_TRAINING = False  # Set to True to resume from checkpoint
    CHECKPOINT_PATH = "checkpoints/best_fuzzy_model.pth"  # Path to checkpoint
    START_EPOCH = 3  # Epoch to start from (after 2 completed epochs)
    TOTAL_EPOCHS = 10  # Total epochs you want (có thể tăng từ 5 lên 10 để train thêm)
    
    # Override config epochs if specified
    if TOTAL_EPOCHS:
        config['num_epochs'] = TOTAL_EPOCHS
    
    print(f"Enhanced Vietnamese VQA Training with Multiple Correct Answers")
    print(f"Using device: {config['device']}")
    
    # List available checkpoints for reference
    available_checkpoints = list_available_checkpoints()
    
    if RESUME_TRAINING:
        print(f"\n🔄 Resume training mode: Starting from epoch {START_EPOCH}")
        print(f"📁 Checkpoint path: {CHECKPOINT_PATH}")
        if CHECKPOINT_PATH not in available_checkpoints and available_checkpoints:
            print(f"⚠️  Warning: Specified checkpoint not found in scan results")
            print(f"   Consider using one of the found checkpoints above")
    else:
        print(f"\n🆕 Fresh training mode: Starting from epoch 1")
    
    # Load and prepare data
    print(f"\nLoading data...")
    df = pd.read_csv(f'{config["text_dir"]}/evaluate_60k_data_balanced_preprocessed.csv')
    df = df.iloc[:5]
    questions = prepare_data_from_dataframe(df)
    
    # Data analysis for multiple answers
    analyze_data_balance(questions)
    
    # Split data
    split_idx = int(0.8 * len(questions))
    train_questions = questions[:split_idx]
    val_questions = questions[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train questions: {len(train_questions):,}")
    print(f"  Validation questions: {len(val_questions):,}")
    
    # Initialize tokenizers and processors
    print(f"\nLoading tokenizers and processors...")
    question_tokenizer = XLMRobertaTokenizer.from_pretrained(config['text_model'])
    answer_tokenizer = T5Tokenizer.from_pretrained(config['decoder_model'], legacy=False)
    clip_processor = CLIPProcessor.from_pretrained(config['vision_model'])

    # Test multiple answer normalization
    print(f"\nTesting multiple answer support...")
    if train_questions and 'all_correct_answers' in train_questions[0]:
        sample_answers = train_questions[0]['all_correct_answers']
        print(f"Sample question: {train_questions[0]['question']}")
        print(f"All correct answers:")
        for i, ans in enumerate(sample_answers, 1):
            normalized = normalize_vietnamese_answer(ans)
            print(f"  {i}. '{ans}' → '{normalized}'")
    
    # Create datasets with multiple answers support
    print(f"\nCreating datasets with multiple correct answers...")
    train_dataset = VietnameseVQADataset(
        train_questions, config['image_dir'], question_tokenizer, 
        answer_tokenizer, clip_processor, config['max_length']
    )
    
    val_dataset = VietnameseVQADataset(
        val_questions, config['image_dir'], question_tokenizer, 
        answer_tokenizer, clip_processor, config['max_length']
    )
    
    # Create data loaders with reduced num_workers to avoid issues
    print(f"\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=0, pin_memory=True if config['device'] == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=0, pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Initialize enhanced model
    print(f"\nInitializing enhanced model...")
    model = ImprovedVietnameseVQAModel(config)
    model = model.to(config['device'])
    
    # Load checkpoint if resuming training
    start_epoch = 1
    best_accuracy = 0.0
    
    if RESUME_TRAINING:
        import os
        if os.path.exists(CHECKPOINT_PATH):
            print(f"\nLoading checkpoint from {CHECKPOINT_PATH}...")
            try:
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=config['device'])
                
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✓ Model state loaded successfully")
                
                # Get training info
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"  ✓ Will resume from epoch {start_epoch}")
                else:
                    start_epoch = START_EPOCH
                    print(f"  ✓ Will start from configured epoch {start_epoch}")
                
                if 'best_accuracy' in checkpoint:
                    best_accuracy = checkpoint['best_accuracy']
                    print(f"  ✓ Previous best accuracy: {best_accuracy:.4f}")
                
                if 'config' in checkpoint:
                    saved_config = checkpoint['config']
                    print(f"  ✓ Checkpoint config loaded")
                    print(f"    - Original batch size: {saved_config.get('batch_size', 'N/A')}")
                    print(f"    - Original learning rates: decoder={saved_config.get('decoder_lr', 'N/A'):.2e}")
                
                print(f"  ✓ Checkpoint loaded successfully!")
                
            except Exception as e:
                print(f"  ✗ Error loading checkpoint: {e}")
                print(f"  Starting fresh training instead...")
                start_epoch = 1
                best_accuracy = 0.0
        else:
            print(f"\n⚠️  Checkpoint file not found: {CHECKPOINT_PATH}")
            print(f"  Starting fresh training instead...")
            start_epoch = 1
            best_accuracy = 0.0
    
    # Update config with start epoch
    config['start_epoch'] = start_epoch
    config['resume_best_accuracy'] = best_accuracy
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nEnhanced Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
    
    # Test model forward pass
    print(f"\nTesting model forward pass...")
    try:
        test_batch = next(iter(train_loader))
        for key, value in test_batch.items():
            if isinstance(value, torch.Tensor):
                test_batch[key] = value.to(config['device'])
        
        with torch.no_grad():
            outputs = model(
                pixel_values=test_batch['pixel_values'][:2],  # Test with 2 samples
                question_input_ids=test_batch['question_input_ids'][:2],
                question_attention_mask=test_batch['question_attention_mask'][:2],
                answer_input_ids=test_batch['answer_input_ids'][:2],
                answer_attention_mask=test_batch['answer_attention_mask'][:2]
            )
            print(f"  ✓ Forward pass successful")
            print(f"  Loss: {outputs.loss.item():.4f}")
            print(f"  Logits shape: {outputs.logits.shape}")
    except Exception as e:
        print(f"  Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test inference mode
    print(f"\nTesting inference mode...")
    try:
        with torch.no_grad():
            generated_ids = model(
                pixel_values=test_batch['pixel_values'][:1],
                question_input_ids=test_batch['question_input_ids'][:1],
                question_attention_mask=test_batch['question_attention_mask'][:1]
            )
            
            pred_text = model.decoder_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            clean_pred_text = model.clean_generated_text(pred_text)
            print(f"  ✓ Inference successful")
            print(f"  Sample prediction raw: '{pred_text}'")
            print(f"  Sample prediction: '{clean_pred_text}'")
            print(f"  Sample ground truth: '{test_batch['answer_text'][0]}'")
    except Exception as e:
        print(f"  Error in inference: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize trainer
    print(f"\nInitializing VQA trainer...")
    trainer = VQATrainer(model, train_loader, val_loader, torch.device(config['device']), config)
    
    # Set resume state if loading from checkpoint
    if RESUME_TRAINING and best_accuracy > 0:
        trainer.best_accuracy = best_accuracy
        print(f"  ✓ Trainer initialized with best accuracy: {best_accuracy:.4f}")
    
    # Start training
    print(f"\n{'='*80}")
    if RESUME_TRAINING:
        print(f"RESUMING ENHANCED TRAINING FROM EPOCH {start_epoch}")
    else:
        print(f"STARTING ENHANCED TRAINING")
    print(f"{'='*80}")
    
    total_epochs = config['num_epochs']
    remaining_epochs = total_epochs - (start_epoch - 1)
    
    print(f"Training configuration:")
    print(f"  Total epochs: {total_epochs}")
    print(f"  Starting from epoch: {start_epoch}")
    print(f"  Remaining epochs: {remaining_epochs}")
    print(f"  Previous best accuracy: {best_accuracy:.4f}" if best_accuracy > 0 else "  No previous best accuracy")
    print(f"  Decoder LR: {config['decoder_lr']:.2e}")
    print(f"  Encoder LR: {config['encoder_lr']:.2e}")
    print(f"  Vision LR: {config['vision_lr']:.2e}")
    print(f"  Label smoothing: {config['label_smoothing']}")
    print(f"  Dropout rate: {config['dropout_rate']}")
    print(f"  Warmup ratio: {config.get('warmup_ratio', 0.1)}")
    print(f"  Data augmentation: {config.get('use_data_augmentation', False)}")
    print(f"  Wandb logging: {config.get('use_wandb', False)}")
    
    try:
        # Calculate remaining epochs and train
        if RESUME_TRAINING:
            remaining_epochs = total_epochs - (start_epoch - 1)
            print(f"\nTraining {remaining_epochs} remaining epochs...")
            
            # Pass the correct start epoch information to trainer
            final_best_accuracy = trainer.train(remaining_epochs, start_epoch=start_epoch)
        else:
            final_best_accuracy = trainer.train(config['num_epochs'], start_epoch=1)
        
        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Final best accuracy achieved: {final_best_accuracy:.4f}")
        if RESUME_TRAINING and best_accuracy > 0:
            improvement = final_best_accuracy - best_accuracy
            print(f"Improvement from resumed training: {improvement:+.4f}")
        print(f"Model and checkpoints saved in current directory")
        print(f"Predictions saved for analysis")
        
    except KeyboardInterrupt:
        print(f"Training interrupted by user")
        print(f"Saving current state...")
        try:
            # Try to save current checkpoint if possible
            current_epoch = getattr(trainer, 'current_epoch', 0)
            trainer.save_checkpoint(current_epoch, {}, is_best=False)
            print(f"Checkpoint saved for epoch {current_epoch}")
        except:
            print(f"Could not save checkpoint")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
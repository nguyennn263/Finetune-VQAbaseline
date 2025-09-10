import torch
import os

def get_improved_config():
    """Improved configuration for VQA with CLIP, XLM-RoBERTa, and mT5"""
    
    # Detect environment
    is_kaggle = os.path.exists('/kaggle')
    
    if is_kaggle:
        # Import Kaggle config if available
        try:
            from .kaggle_config import get_kaggle_config
            return get_kaggle_config()
        except ImportError:
            pass  # Fall back to default config
    
    # Default configuration
    return {
        'vision_model': 'openai/clip-vit-base-patch32',
        'text_model': 'xlm-roberta-base',
        'decoder_model': 'google/mt5-small',
        'hidden_dim': 768,  # Standard dimension for base models
        'max_length': 128,
        'batch_size': 1,  # Reduced for GPU memory
        'num_epochs': 3,
        'image_dir': '/home/nguyennn263/Documents/Thesis/Fintune/Dataset/images/images',
        'text_dir': '/home/nguyennn263/Documents/Thesis/Fintune/Dataset/text/text',
        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'device': 'cpu',
        
        # cxmt5 specific configurations
        'use_vqkd': True,  # Enable VQ-KD Visual Tokenizer
        'visual_vocab_size': 8192,  # Codebook size
        'num_multiway_layers': 6,  # Number of Multiway Transformer layers
        
        # Unified Masked Data Modeling
        'use_unified_masking': True,
        'text_mask_ratio': 0.15,  # 15% for monomodal text
        'multimodal_text_mask_ratio': 0.50,  # 50% for multimodal text
        'vision_mask_ratio': 0.40,  # 40% for image patches
        
        # Staged training configuration
        'stage1_epochs': 4,  # Freeze encoders
        'stage2_epochs':6,  # Partial unfreeze
        
        # Different learning rates
        'decoder_lr': 1e-4,
        'encoder_lr': 1e-5,
        'vision_lr': 5e-6,
        
        # Enhanced scheduler configuration
        'warmup_ratio': 0.1,
        'scheduler_type': 'linear_decay_with_warmup',
        'weight_decay': 0.01,
        
        # Enhanced regularization
        'label_smoothing': 0.1,
        'dropout_rate': 0.2,
        
        # Unfreezing strategy
        'unfreeze_last_n_layers': 4,
        
        # Data augmentation
        'use_data_augmentation': True,
        'augment_ratio': 0.2,
        
        # Logging and checkpoints
        'use_wandb': False,
        'project_name': 'CXMT5-Vietnamese-VQA',
        'save_every_n_epochs': 1,
        'keep_last_n_checkpoints': 5,
        
        # Enhanced evaluation
        'evaluate_every_n_steps': 5000,
        'save_predictions': True,
        'calculate_bleu_rouge': True,
        'calculate_cider': True,
        
        'version': "1.1",
    }
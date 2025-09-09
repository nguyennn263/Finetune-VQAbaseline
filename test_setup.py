#!/usr/bin/env python3
"""
Test script để kiểm tra setup và khả năng chạy model VQA
"""

import sys
import os

def test_imports():
    """Test import các thư viện cần thiết"""
    print("Testing imports...")
    
    try:
        import transformers
        print(f"✓ transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"✗ torch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✓ pandas version: {pd.__version__}")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
        
    try:
        from PIL import Image
        print("✓ PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"✗ PIL import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test loading các models cần thiết"""
    print("\nTesting model loading...")
    
    try:
        from transformers import AutoTokenizer, CLIPProcessor, T5ForConditionalGeneration
        from transformers import CLIPVisionModel, XLMRobertaModel
        
        print("Testing XLM-RoBERTa tokenizer...")
        text_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        print("✓ XLM-RoBERTa tokenizer loaded")
        
        print("Testing mT5 tokenizer...")
        mt5_tokenizer = AutoTokenizer.from_pretrained('google/mt5-base', use_fast=False)
        print("✓ mT5 tokenizer loaded")
        
        print("Testing CLIP processor...")
        clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        print("✓ CLIP processor loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_config():
    """Test config loading"""
    print("\nTesting config...")
    
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from cxmt5.config import get_improved_config
        
        config = get_improved_config()
        print("✓ Config loaded successfully")
        print(f"  Vision model: {config['vision_model']}")
        print(f"  Text model: {config['text_model']}")
        print(f"  Decoder model: {config['decoder_model']}")
        print(f"  Device: {config['device']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_data_loader():
    """Test data loader"""
    print("\nTesting data loader...")
    
    try:
        from data_loader import create_sample_data, validate_vqa_data
        
        df = create_sample_data()
        print(f"✓ Sample data created: {len(df)} samples")
        
        validation_results = validate_vqa_data(df)
        print("✓ Data validation successful")
        print(f"  Total samples: {validation_results['total_samples']}")
        print(f"  Avg answers per question: {validation_results['statistics']['avg_answers_per_question']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return False

def test_model_initialization():
    """Test khởi tạo model VQA"""
    print("\nTesting VQA model initialization...")
    
    try:
        from cxmt5.config import get_improved_config
        from cxmt5.cxmt5 import VietnameseVQAModel
        
        config = get_improved_config()
        
        print("Initializing VQA model...")
        model = VietnameseVQAModel(config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("✓ VQA model initialized successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass của model"""
    print("\nTesting forward pass...")
    
    try:
        from cxmt5.config import get_improved_config
        from cxmt5.cxmt5 import VietnameseVQAModel
        from transformers import AutoTokenizer, CLIPProcessor
        from PIL import Image
        import torch
        
        config = get_improved_config()
        
        # Khởi tạo model và tokenizers
        model = VietnameseVQAModel(config)
        question_tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
        clip_processor = CLIPProcessor.from_pretrained(config['vision_model'])
        
        # Tạo dummy input
        dummy_image = Image.new('RGB', (224, 224), color='white')
        dummy_question = "Trong ảnh có gì?"
        
        # Process inputs
        image_inputs = clip_processor(images=dummy_image, return_tensors="pt")
        question_inputs = question_tokenizer(
            dummy_question, 
            max_length=128, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Forward pass (inference mode)
        with torch.no_grad():
            outputs = model(
                pixel_values=image_inputs['pixel_values'],
                question_input_ids=question_inputs['input_ids'],
                question_attention_mask=question_inputs['attention_mask']
            )
        
        print("✓ Forward pass successful")
        print(f"  Output shape: {outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def main():
    """Chạy tất cả tests"""
    print("VQA Model Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("Config Test", test_config),
        ("Data Loader Test", test_data_loader),
        ("Model Initialization Test", test_model_initialization),
        ("Forward Pass Test", test_forward_pass)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 20}")
        print(f"Running: {test_name}")
        print(f"{'-' * 20}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("Test Summary")
    print(f"{'=' * 50}")
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<30} {status}")
        if success:
            passed += 1
    
    print(f"\n{passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your setup is ready for VQA training.")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please fix the issues before training.")
        print("\nCommon solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check internet connection for model downloads")
        print("- Ensure sufficient GPU memory if using CUDA")

if __name__ == "__main__":
    main()

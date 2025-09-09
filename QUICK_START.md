# Quick Start Guide - Vietnamese VQA

## Bước 1: Kiểm tra setup

```bash
python test_setup.py
```

## Bước 2: Chuẩn bị dữ liệu của bạn

### Định dạng dữ liệu cần có:

**CSV format** (khuyến nghị):
```csv
image_name,question,answers
image1.jpg,"Trong ảnh có gì?","một chiếc xe|xe hơi|ô tô"
image2.jpg,"Màu sắc chủ đạo là gì?","màu xanh|xanh lá"
image3.jpg,"Có bao nhiêu người?","hai người|2 người|2"
```

**JSON format**:
```json
[
  {
    "image_name": "image1.jpg",
    "question": "Trong ảnh có gì?",
    "answers": ["một chiếc xe", "xe hơi", "ô tô"]
  }
]
```

### Cấu trúc thư mục:
```
your_project/
├── data/
│   ├── images/           # Thư mục chứa ảnh
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── vqa_data.csv      # File dữ liệu VQA
├── vqa_main.py
└── ...
```

## Bước 3: Chỉnh sửa config

Mở `cxmt5/config.py` và sửa:

```python
def get_improved_config():
    return {
        # ... other configs
        'image_dir': '/path/to/your/images',  # ← Thay đổi đường dẫn này
        'batch_size': 8,  # Giảm nếu bị out of memory
        'num_epochs': 5,  # Số epoch training
        # ...
    }
```

## Bước 4: Sửa data loading

Mở `vqa_main.py` và sửa hàm `load_and_prepare_data()`:

```python
def load_and_prepare_data(config):
    """Load and prepare VQA dataset"""
    
    print("Loading VQA dataset...")
    
    # THAY ĐỔI DÒNG NÀY - Load dữ liệu thật của bạn
    df = pd.read_csv('/path/to/your/vqa_data.csv')  # ← Đường dẫn file dữ liệu
    
    # Convert to questions format
    questions = prepare_data_from_dataframe(df)
    
    return questions
```

## Bước 5: Chạy training

```bash
python vqa_main.py
```

## Models được sử dụng:

1. **Vision Encoder**: CLIP ViT-Base-Patch32
   - Xử lý hình ảnh đầu vào
   - Extract visual features

2. **Text Encoder**: XLM-RoBERTa-Base  
   - Encode câu hỏi tiếng Việt
   - Multilingual support

3. **Text Decoder**: mT5-Base
   - Generate câu trả lời
   - Multilingual text generation

## Kết quả training:

Sau khi training sẽ có các file:
- `best_vqa_model.pth` - Model tốt nhất
- `best_vqa_results.json` - Kết quả chi tiết
- `final_evaluation_results.json` - Đánh giá cuối cùng

## Evaluation metrics:

- **VQA Score**: Metric chuẩn cho VQA (0-1)
- **Multi Exact Accuracy**: Độ chính xác chính xác 100%
- **Multi Fuzzy Accuracy**: Độ chính xác với tolerance
- **Multi Token F1**: F1 score ở mức token
- **Multi BLEU**: BLEU score

## Troubleshooting:

### Out of Memory:
```python
# Trong config.py
'batch_size': 4,  # Giảm batch size
```

### Data không load được:
- Kiểm tra đường dẫn file
- Kiểm tra format CSV/JSON
- Đảm bảo có đủ cột: image_name, question, answers

### Images không tìm thấy:
- Kiểm tra `image_dir` trong config
- Đảm bảo tên file trong data khớp với file thật

## Ví dụ complete workflow:

```python
# 1. Kiểm tra setup
python test_setup.py

# 2. Load và kiểm tra data
from data_loader import load_vqa_data, validate_vqa_data
df = load_vqa_data('your_data.csv')
print(validate_vqa_data(df))

# 3. Chạy training
python vqa_main.py

# 4. Evaluate results
import json
with open('best_vqa_results.json', 'r') as f:
    results = json.load(f)
    print(f"Best VQA Score: {results['metrics']['vqa_score']:.4f}")
```

## Tips để có kết quả tốt:

1. **Dữ liệu tốt**: Đảm bảo multiple answers phong phú và chính xác
2. **Augmentation**: Bật data augmentation trong config
3. **Learning rate**: Thử các learning rate khác nhau
4. **Epochs**: Training đủ epochs (10-20)
5. **Evaluation**: Kiểm tra metrics trong quá trình training

Chúc bạn training thành công! 🚀

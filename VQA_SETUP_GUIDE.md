# Vietnamese VQA with CLIP + XLM-RoBERTa + mT5

Dự án này triển khai một hệ thống Visual Question Answering (VQA) cho tiếng Việt sử dụng:
- **Vision Model**: CLIP ViT-Base-Patch32 (openai/clip-vit-base-patch32)
- **Text Encoder**: XLM-RoBERTa-Base (xlm-roberta-base) 
- **Text Decoder**: mT5-Base (google/mt5-base)

## Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu

Dữ liệu VQA của bạn cần có định dạng:
- **1 ảnh** - **1 câu hỏi** - **nhiều câu trả lời**

#### Định dạng CSV:
```csv
image_name,question,answers
image1.jpg,"Trong ảnh có gì?","một chiếc xe|xe hơi|ô tô|xe cộ"
image2.jpg,"Màu sắc chủ đạo là gì?","màu xanh|xanh lá cây|xanh"
```

#### Định dạng JSON:
```json
[
  {
    "image_name": "image1.jpg",
    "question": "Trong ảnh có gì?",
    "answers": ["một chiếc xe", "xe hơi", "ô tô", "xe cộ"]
  }
]
```

## Sử dụng

### 1. Chuẩn bị dữ liệu

Chỉnh sửa file `data_loader.py` để load dữ liệu của bạn:

```python
def load_your_data(data_path: str) -> pd.DataFrame:
    # Thay đổi đường dẫn và cách load dữ liệu của bạn
    df = pd.read_csv(data_path)  # hoặc pd.read_json(data_path)
    
    # Đảm bảo có các cột: image_name, question, answers
    return standardize_vqa_dataframe(df)
```

### 2. Cấu hình model

Chỉnh sửa `bartphobeit/config.py` nếu cần:

```python
def get_improved_config():
    return {
        'vision_model': 'openai/clip-vit-base-patch32',
        'text_model': 'xlm-roberta-base', 
        'decoder_model': 'google/mt5-base',
        'image_dir': '/path/to/your/images',  # Thay đổi đường dẫn ảnh
        'batch_size': 16,
        'num_epochs': 10,
        # ... other configs
    }
```

### 3. Chạy training

```bash
python vqa_main.py
```

Hoặc chạy trong notebook:

```python
from vqa_main import main
main()
```

## Cấu trúc dữ liệu mong đợi

### Cấu trúc thư mục:
```
your_project/
├── data/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── vqa_data.csv  # hoặc .json
├── bartphobeit/
├── vqa_main.py
└── data_loader.py
```

### Định dạng dữ liệu:

**Bắt buộc các cột:**
- `image_name`: tên file ảnh (VD: "image1.jpg")
- `question`: câu hỏi (VD: "Trong ảnh có gì?")
- `answers`: nhiều câu trả lời đúng (VD: ["xe hơi", "ô tô", "xe cộ"])

**Các cách định dạng answers được hỗ trợ:**
```python
# Dạng list trong JSON
"answers": ["câu trả lời 1", "câu trả lời 2"]

# Dạng string với separator trong CSV
"answers": "câu trả lời 1|câu trả lời 2|câu trả lời 3"
# hoặc 
"answers": "câu trả lời 1, câu trả lời 2, câu trả lời 3"
```

## Ví dụ sử dụng với dữ liệu của bạn

### 1. Load và kiểm tra dữ liệu

```python
from data_loader import load_vqa_data, validate_vqa_data

# Load dữ liệu
df = load_vqa_data('path/to/your/data.csv')

# Kiểm tra dữ liệu
validation_results = validate_vqa_data(df)
print(validation_results)
```

### 2. Training script tùy chỉnh

```python
import torch
from bartphobeit.config import get_improved_config
from bartphobeit.BARTphoBEIT import VietnameseVQAModel, VQATrainer
from data_loader import load_your_data

# Load config và dữ liệu
config = get_improved_config()
config['image_dir'] = '/path/to/your/images'

# Load dữ liệu của bạn
df = load_your_data('/path/to/your/data.csv')

# Khởi tạo model và training
# ... (xem vqa_main.py để có example đầy đủ)
```

## Các tính năng chính

### 1. Multimodal Fusion
- Kết hợp features từ CLIP vision encoder và XLM-RoBERTa text encoder
- Sử dụng Multi-head Attention để fusion

### 2. Multiple Answer Support  
- Hỗ trợ đánh giá với nhiều câu trả lời đúng
- Tính toán VQA Score theo chuẩn VQA dataset

### 3. Comprehensive Evaluation
- VQA Score (chuẩn)
- Multi Exact Accuracy
- Multi Fuzzy Accuracy  
- Multi Token F1
- BLEU Score

### 4. Model Checkpointing
- Tự động lưu model tốt nhất
- Hỗ trợ resume training

## Kết quả Training

Sau khi training, bạn sẽ có:

```
best_vqa_model.pth          # Model với VQA score cao nhất
best_fuzzy_model.pth        # Model với fuzzy accuracy cao nhất
best_vqa_results.json       # Kết quả chi tiết của model tốt nhất
final_evaluation_results.json # Kết quả cuối cùng
```

## Đánh giá Model

```python
from bartphobeit.BARTphoBEIT import VietnameseVQAModel
from bartphobeit.model import compute_metrics

# Load model
model = VietnameseVQAModel(config)
model.load_state_dict(torch.load('best_vqa_model.pth'))

# Evaluate
predictions = ["dự đoán 1", "dự đoán 2", ...]
ground_truths = [["đáp án 1", "đáp án 2"], ["đáp án 1"], ...]
metrics = compute_metrics(predictions, ground_truths, tokenizer)
```

## Troubleshooting

### 1. GPU Memory Issues
- Giảm `batch_size` trong config
- Sử dụng gradient accumulation

### 2. Data Loading Issues
- Kiểm tra đường dẫn ảnh trong `image_dir`
- Đảm bảo format dữ liệu đúng

### 3. Model Performance
- Tăng số epoch training
- Điều chỉnh learning rate
- Thử unfreeze encoder layers

## Support

Nếu gặp vấn đề, hãy kiểm tra:
1. Format dữ liệu đúng chuẩn
2. Đường dẫn ảnh chính xác  
3. GPU memory đủ cho batch size
4. Các dependency đã cài đầy đủ

# Kaggle Deployment Guide for VQA Project

## Bước 1: Chuẩn bị Dataset

### 1.1 Upload Dataset lên Kaggle
1. Đăng nhập vào Kaggle.com
2. Đi tới "Datasets" → "New Dataset"
3. Upload các file sau:
   - CSV file chứa câu hỏi và câu trả lời
   - Folder `images/` chứa tất cả hình ảnh
   - Metadata files (nếu có)

### 1.2 Cấu trúc Dataset trên Kaggle
```
your-vqa-dataset/
├── train.csv
├── test.csv  
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── metadata.json (optional)
```

## Bước 2: Tạo Kaggle Notebook

### 2.1 Tạo Notebook mới
1. Đi tới "Code" → "New Notebook"
2. Chọn "GPU P100" hoặc "GPU T4" (miễn phí)
3. Enable Internet trong Settings

### 2.2 Upload Code
Có 2 cách:

**Cách 1: Upload thư mục (Recommended)**
1. Zip toàn bộ project: `zip -r vqa-project.zip .`
2. Upload zip file vào Kaggle Dataset
3. Add dataset vào notebook

**Cách 2: Copy-paste code**
- Copy từng file Python vào notebook cells

## Bước 3: Kaggle Notebook Setup

### 3.1 Cell 1: Install Dependencies
```python
# Install requirements
!pip install -r /kaggle/input/your-vqa-project/requirements.txt

# Or install individually
!pip install transformers torch torchvision accelerate rouge_score nltk sentencepiece
```

### 3.2 Cell 2: Setup Environment
```python
import sys
sys.path.append('/kaggle/input/your-vqa-project')

# Import setup
from kaggle_setup import get_kaggle_config, print_system_info
print_system_info()
```

### 3.3 Cell 3: Configure Paths
```python
# Update paths in kaggle_config.py
# Change 'your-vqa-dataset' to your actual dataset name
base_path = '/kaggle/input/your-actual-dataset-name'
```

### 3.4 Cell 4: Run Training
```python
# Run the main training script
exec(open('/kaggle/input/your-vqa-project/kaggle_train.py').read())
```

## Bước 4: GPU Optimization Settings

### 4.1 Memory Management
- Batch size sẽ được tự động điều chỉnh dựa trên GPU memory
- Mixed precision training được enable tự động
- Gradient checkpointing để tiết kiệm memory

### 4.2 Time Management
- Training được giới hạn 8.5 giờ (để có buffer cho 9h limit của Kaggle)
- Auto-save checkpoints mỗi 600 steps
- Early stopping sau 3 epochs không cải thiện

### 4.3 Performance Monitoring
```python
# Thêm vào notebook để monitor GPU
import torch
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## Bước 5: Kaggle-specific Configurations

### 5.1 Data Loading
- `num_workers=2` (Kaggle có CPU cores hạn chế)
- `pin_memory=True` cho faster GPU transfer
- `persistent_workers=True` để tránh reload overhead

### 5.2 Model Saving
- Checkpoints được save vào `/kaggle/working/checkpoints/`
- Final model được save vào `/kaggle/working/final_model/`
- Training logs trong `/kaggle/working/training_info.txt`

### 5.3 Error Handling
- Tự động save error logs vào `/kaggle/working/error_log.txt`
- Checkpoint recovery nếu training bị interrupt

## Bước 6: Dataset Names to Update

Cần thay đổi trong các file sau:

### 6.1 `kaggle_config.py`
```python
base_path = '/kaggle/input/YOUR-DATASET-NAME'  # Thay YOUR-DATASET-NAME
```

### 6.2 `kaggle_setup.py`
```python
'data_dir': f"{kaggle_input}/YOUR-DATASET-NAME"  # Thay YOUR-DATASET-NAME
```

## Bước 7: Expected Training Time

- **GPU T4**: ~6-8 giờ cho 8 epochs
- **GPU P100**: ~4-6 giờ cho 8 epochs
- **CPU**: Không khuyến khích (quá chậm)

## Bước 8: Tips for Success

### 8.1 Monitoring
- Thêm print statements để theo dõi progress
- Use tqdm progress bars
- Monitor GPU memory usage

### 8.2 Debugging
- Test với 1-2 epochs trước khi chạy full training
- Kiểm tra data loading trước
- Verify model architecture với dummy input

### 8.3 Backup Strategy
- Commit notebook thường xuyên
- Save intermediate checkpoints
- Export final model về local nếu cần

## Troubleshooting

### Common Issues:
1. **Out of Memory**: Giảm batch_size trong config
2. **Time Limit**: Giảm num_epochs hoặc tăng evaluate_every_n_steps
3. **Data Path Error**: Kiểm tra dataset name và structure
4. **Import Error**: Đảm bảo sys.path được setup đúng

### Performance Tips:
1. Sử dụng mixed precision training
2. Enable gradient checkpointing nếu memory thấp
3. Optimize data loading với appropriate num_workers

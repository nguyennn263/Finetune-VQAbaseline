#!/bin/bash

# Kaggle Preparation Script
# This script helps prepare your VQA project for Kaggle deployment

echo "🚀 Preparing VQA Project for Kaggle..."

# Create deployment folder
mkdir -p kaggle_deployment
cd kaggle_deployment

# Copy essential files
echo "📁 Copying project files..."
cp -r ../cxmt5 .
cp -r ../evaluation .
cp ../vqa_main.py .
cp ../data_loader.py .
cp ../data_downloader.py .
cp ../requirements.txt .
cp ../kaggle_setup.py .
cp ../kaggle_train.py .
cp ../KAGGLE_GUIDE.md .
cp ../README.md .

# Remove cache files
echo "🧹 Cleaning cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true

# Create kaggle metadata
echo "📝 Creating Kaggle metadata..."
cat > dataset-metadata.json << EOF
{
  "title": "VQA Baseline Model",
  "id": "your-username/vqa-baseline-model",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ],
  "keywords": [
    "computer vision",
    "natural language processing",
    "vqa",
    "transformers",
    "multimodal"
  ],
  "description": "A baseline VQA model using CLIP, XLM-RoBERTa, and mT5 for visual question answering tasks"
}
EOF

# Create simple notebook template
echo "📓 Creating Kaggle notebook template..."
cat > kaggle_notebook.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# VQA Training on Kaggle\n",
    "\n",
    "This notebook trains a Visual Question Answering model using GPU acceleration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install requirements\n",
    "!pip install transformers torch torchvision accelerate rouge_score nltk sentencepiece"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setup paths\n",
    "import sys\n",
    "sys.path.append('/kaggle/input/vqa-baseline-model')\n",
    "\n",
    "# Print system info\n",
    "from cxmt5.kaggle_config import print_system_info\n",
    "print_system_info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# IMPORTANT: Update dataset name here\n",
    "# Change 'your-vqa-dataset' to your actual dataset name\n",
    "DATASET_NAME = 'your-vqa-dataset'\n",
    "\n",
    "# Verify dataset exists\n",
    "import os\n",
    "dataset_path = f'/kaggle/input/{DATASET_NAME}'\n",
    "print(f\"Dataset path: {dataset_path}\")\n",
    "print(f\"Dataset exists: {os.path.exists(dataset_path)}\")\n",
    "if os.path.exists(dataset_path):\n",
    "    print(\"Dataset contents:\")\n",
    "    for item in os.listdir(dataset_path):\n",
    "        print(f\"  - {item}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run training\n",
    "exec(open('/kaggle/input/vqa-baseline-model/kaggle_train.py').read())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Create zip file for upload
echo "📦 Creating zip file for upload..."
cd ..
zip -r vqa-kaggle-deployment.zip kaggle_deployment/ -x "*.pyc" "*/__pycache__/*"

echo ""
echo "✅ Kaggle deployment prepared!"
echo ""
echo "📋 Next steps:"
echo "1. Upload vqa-kaggle-deployment.zip to Kaggle as a dataset"
echo "2. Upload your VQA dataset (images + CSV) to Kaggle"
echo "3. Create a new Kaggle notebook with GPU enabled"
echo "4. Add both datasets to your notebook"
echo "5. Update dataset names in the notebook"
echo "6. Run the training!"
echo ""
echo "📖 Read KAGGLE_GUIDE.md for detailed instructions"
echo ""
echo "🎯 Files ready in: vqa-kaggle-deployment.zip"

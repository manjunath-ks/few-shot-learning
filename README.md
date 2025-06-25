# Few-Shot Learning Image Classifier

A production-ready few-shot learning system that can learn to classify images with just 3-10 examples per class.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Training Data
Place your images in the appropriate class folders:
```
data/train/
├── class1/    # Replace with your class name
├── class2/    # Replace with your class name
└── class3/    # Replace with your class name
```

### 3. Train the Model
```bash
python train.py
```

### 4. Make Predictions
```bash
python predict.py path/to/image.jpg
```

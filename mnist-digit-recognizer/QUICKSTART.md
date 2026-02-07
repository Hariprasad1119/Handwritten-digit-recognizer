# 🚀 Quick Start Guide

## Get Up and Running in 5 Minutes!

### Step 1: Setup Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-digit-recognizer.git
cd mnist-digit-recognizer

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```

Wait 5-10 minutes while the model trains. You'll see:
- Training progress with accuracy/loss metrics
- Final test accuracy (~99%+)
- Saved model in `models/` folder
- Visualizations in `results/` folder

### Step 3: Make Predictions
```bash
# Interactive mode
python predict.py

# Or predict a specific image
python predict.py path/to/image.png
```

## 📸 Testing with Your Own Images

1. Create/find an image with a handwritten digit
2. Save it as PNG or JPG
3. Run: `python predict.py your_image.png`

**Tips:**
- White digit on black background works best
- Single digit per image
- Clear, centered digit

## 🎯 Expected Results

After training:
- **Accuracy:** 99%+ on test set
- **Model Size:** ~900 KB
- **Inference Time:** <10ms per image

## ❓ Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'tensorflow'`
- **Solution:** `pip install tensorflow`

**Issue:** Model not found error when predicting
- **Solution:** Train the model first with `python train_model.py`

**Issue:** Low accuracy on custom images
- **Solution:** Ensure image has white digit on black background, similar to MNIST format

## 🔗 Next Steps

- Modify the CNN architecture in `train_model.py`
- Try different hyperparameters (learning rate, dropout, etc.)
- Add data augmentation
- Deploy as a web app!

---

**Need help?** Open an issue on GitHub!

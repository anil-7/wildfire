# 🔥 Quick Start Guide - Simplified Workflow

## 📋 Prerequisites
1. ✅ Kaggle credentials (`config/kaggle.json`)
2. ✅ Groq API key (in `config/.env`)

---

## 🚀 ONE-TIME SETUP (Do Once)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Datasets
```bash
python src/data_loader/download_datasets.py
```
Choose option 1 to download all wildfire datasets from Kaggle.

### Step 3: Preprocess Data
```bash
python src/preprocessing/preprocess_data.py
```
This organizes images into train/val/test splits.

### Step 4: Train Models
```bash
python src/training/train_detection_model.py
```
Choose option 1 for single model (faster) or option 2 for ensemble (more accurate).

⏱️ This takes 1-2 hours for single model, 5-8 hours for all models.

---

## 🎯 DAILY USAGE (Every Time)

### Just run one command:
```bash
streamlit run app.py
```

That's it! The UI opens in your browser automatically.

---

## 🎨 Using the Web Interface

1. **🔍 Fire Detection Tab**
   - Upload wildfire images
   - Get instant AI detection results
   - View confidence scores
   - Get AI-powered recommendations

2. **📊 Analytics Tab**
   - View detection statistics
   - See trends and charts
   - Export data

3. **📄 Reports Tab**
   - View fire alerts
   - Generate emergency reports
   - Download reports

4. **ℹ️ About Tab**
   - System information
   - Documentation links

---

## 📝 Summary

**One-Time Setup:**
```bash
pip install -r requirements.txt
python src/data_loader/download_datasets.py
python src/preprocessing/preprocess_data.py
python src/training/train_detection_model.py
```

**Daily Usage:**
```bash
streamlit run app.py
```

That's all you need! 🎉

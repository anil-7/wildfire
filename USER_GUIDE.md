# AI-Integrated Smart Wildfire Management System
# User Guide

## Overview
This system uses advanced AI to detect wildfires, predict their spread, and provide emergency coordination tools.

## Quick Start

### 1. Initial Setup
```bash
python setup.py
```
This will guide you through the initial configuration.

### 2. Configure Credentials

#### Kaggle API (Required for dataset download)
1. Go to https://www.kaggle.com/
2. Navigate to Account Settings → API
3. Click "Create New Token"
4. Save the downloaded `kaggle.json` to `config/kaggle.json`

#### Groq API (Optional but recommended for AI insights)
1. Visit https://console.groq.com/
2. Create an account and get an API key
3. Add to `config/.env`:
   ```
   GROQ_API_KEY=your_actual_api_key_here
   ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Datasets
```bash
python src/data_loader/download_datasets.py
```

### 5. Preprocess Data
```bash
python src/preprocessing/preprocess_data.py
```

### 6. Train Models
```bash
python src/training/train_detection_model.py
```

Choose option 1 for single model (faster) or option 2 for ensemble (higher accuracy).

## Usage

### Web Dashboard (Recommended)
```bash
python app/main.py --dashboard
```

This launches an interactive web interface with:
- Early wildfire detection
- Spread prediction
- Emergency coordination tools
- Real-time analytics

### Command Line Interface

#### Analyze Single Image
```bash
python app/main.py --image path/to/image.jpg
```

#### Analyze Video
```bash
python app/main.py --video path/to/video.mp4
```

#### Batch Process Directory
```bash
python app/main.py --directory path/to/images/
```

#### Generate Emergency Report
```bash
python app/main.py --image fire.jpg --report
```

### Advanced Options
```bash
# Disable Groq AI insights
python app/main.py --image photo.jpg --no-groq

# Custom frame interval for video
python app/main.py --video fire.mp4 --frame-interval 15
```

## Features

### 🔍 Early Detection
- Hybrid CNN models (EfficientNet, ResNet, Inception, Custom, Attention-based)
- Real-time confidence scores
- High accuracy detection

### 📈 Spread Prediction
- Temporal-spatial modeling
- Risk level assessment
- Critical zone identification

### 🤖 AI Insights (Groq Integration)
- Intelligent risk assessment
- Actionable recommendations
- Emergency coordination reports

### 📊 Visualizations
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Scatter plots
- Training history plots
- Prediction visualizations

### 🚨 Emergency Coordination
- Real-time dashboard
- Alert management
- Report generation
- Emergency contacts

## Project Structure

```
wild fires/
├── app/                    # Main application
│   └── main.py            # CLI entry point
├── config/                # Configuration files
│   ├── config.py         # Main config
│   ├── .env              # Environment variables
│   └── kaggle.json       # Kaggle credentials
├── data/                  # Datasets
│   ├── raw/              # Downloaded data
│   └── processed/        # Preprocessed data
├── models/               # Trained models
│   ├── detection/        # Detection models
│   └── prediction/       # Prediction models
├── src/                  # Source code
│   ├── data_loader/      # Dataset download
│   ├── preprocessing/    # Data preprocessing
│   ├── models/           # Model architectures
│   ├── training/         # Training scripts
│   ├── prediction/       # Inference
│   ├── visualization/    # Visualizations
│   ├── groq_integration/ # Groq AI
│   └── coordination/     # Dashboard
├── outputs/              # Results
│   ├── visualizations/   # Plots and charts
│   ├── predictions/      # Prediction results
│   └── reports/          # Emergency reports
└── README.md
```

## Model Information

### Detection Models
1. **EfficientNet** (Recommended) - Best balance of speed and accuracy
2. **ResNet** - Robust feature extraction
3. **Inception** - Multi-scale feature detection
4. **Custom CNN** - Optimized for wildfire patterns
5. **Attention CNN** - Focus on relevant regions

### Ensemble Approach
All models can be combined for maximum accuracy through weighted averaging.

## Troubleshooting

### Model Not Found
- Ensure you've trained models: `python src/training/train_detection_model.py`
- Check `models/detection/` directory for .h5 files

### Kaggle Download Fails
- Verify `kaggle.json` is in `config/` directory
- Check file permissions (Unix: `chmod 600 config/kaggle.json`)
- Ensure internet connection

### Groq AI Not Working
- Verify API key in `config/.env`
- Check API key is valid at https://console.groq.com/
- System works without Groq, but with reduced insights

### Out of Memory
- Reduce batch size in `config/.env`: `BATCH_SIZE=16`
- Reduce image size: `MAX_IMAGE_SIZE=256`
- Use single model instead of ensemble

## Performance Tips

1. **For Best Accuracy**: Train all models and use ensemble
2. **For Speed**: Use single EfficientNet model
3. **For Large Datasets**: Increase batch size if you have enough RAM
4. **For Video**: Increase frame interval to process fewer frames

## Phase 4: Collaboration & Scaling

### Integration with Disaster Management Agencies
The system provides:
- RESTful API endpoints (future enhancement)
- Standardized report formats
- Real-time dashboard access
- Emergency alert system
- Data export capabilities

### Deployment Considerations
- Docker containerization (future enhancement)
- Cloud deployment (AWS, Azure, GCP)
- Mobile app integration
- Multi-user access control

## Support

For issues or questions:
1. Check this user guide
2. Review README.md
3. Check error messages in console
4. Verify all configuration files

## License
MIT License - See LICENSE file for details

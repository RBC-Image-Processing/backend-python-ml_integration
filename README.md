# RBC MIDaP - Python ML Backend

## Project Overview

This is the machine learning backend for the RBC MIDaP (Medical Image Diagnostic and Processing) platform. It provides a FastAPI-based REST API that:

1. Accepts chest X-ray images (DICOM or standard formats)
2. Runs a trained CNN model (ResNet) for **pneumonia detection**
3. Generates clinical interpretations using AI services (Google Gemini, Groq, or OpenAI)

The Node.js backend sends DICOM images to this service and receives predictions with confidence scores and clinical interpretations.

## Architecture

```
Node.js Backend
    |
    v  (POST /predict_with_interpretation)
FastAPI Server (this repo)
    |
    ├── Image Processing (CLAHE for DICOM, standard transforms for JPEG/PNG)
    ├── CNN Model (ResNet101) → prediction + confidence score
    └── AI Interpretation (Gemini / Groq / OpenAI) → clinical text
```

## Prerequisites

- **Python** 3.8+
- **PyTorch** 2.4+ (with CUDA for GPU acceleration, optional)
- **Trained model file** (`cnn.pth`) in the `output/` directory

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RBC-Image-Processing/backend-python-ml_integration.git
cd backend-python-ml_integration
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys (see [Environment Variables](#environment-variables) below).

### 5. Place the Trained Model

Place your trained model file at `output/cnn.pth`. See [Model Setup](#model-setup) for details.

### 6. Start the Server

```bash
python api/main.py
```

The server starts on `http://0.0.0.0:8000`.

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for AI interpretation | Yes (for Gemini endpoint) |
| `GROQ_API_KEY` | Groq API key for AI interpretation | Yes (for Groq endpoint) |
| `OPEN_AI_API_KEY` | OpenAI API key for AI interpretation | Yes (for OpenAI endpoint) |
| `KAGGLE_USERNAME` | Kaggle username for remote training | Optional |
| `KAGGLE_KEY` | Kaggle API key | Optional |
| `NOTEBOOK_PATH` | Path to Kaggle training notebook | Optional |
| `KAGGLE_KERNEL_NAME` | Kaggle kernel name | Optional |
| `KAGGLE_DATASET_PATH` | Kaggle dataset identifier | Optional |
| `MODEL_FILENAME` | Output model filename | Optional |
| `MACHINE_TYPE` | GPU or TPU for Kaggle training | Optional |
| `POLLING_INTERVAL` | Kaggle polling interval in seconds | Optional |

> **Note:** You need at least one AI API key (Gemini, Groq, or OpenAI) for the interpretation endpoints. The `/predict` endpoint works without any API key.

## Model Setup

The CNN model file must be placed at `output/cnn.pth`. This file is **gitignored** (91MB) and must be obtained separately.

### Option 1: Pull from Kaggle (Recommended)

If the model has already been trained on Kaggle, pull it directly:

```bash
python scripts/pull_trained_model_from_kaggle.py
```

This downloads the trained model from the team's Kaggle kernel output and saves it to `output/cnn.pth`.

### Option 2: Get from a Team Member

Ask a team member who has the trained model to share their `cnn.pth` file. Place it at `output/cnn.pth`.

### Option 3: Train on Kaggle from Scratch

1. Set up Kaggle credentials in `.env`
2. Generate a training notebook:
   ```bash
   python scripts/generators/generate_notebook_v3.py
   ```
3. Push to Kaggle:
   ```bash
   python scripts/send_notebook_to_kaggle.py
   ```
4. Wait for training to complete, then pull the model:
   ```bash
   python scripts/pull_trained_model_from_kaggle.py
   ```

### Model Architecture

- **Base model:** ResNet101 (also supports ResNet50, ResNet18)
- **Classification:** Binary (NORMAL / PNEUMONIA)
- **Input size:** 224x224 RGB images
- **Normalization:** ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Running the Server

```bash
# Standard startup
python api/main.py

# Alternative with uvicorn directly
uvicorn api.api_endpoints:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload during development.

## API Endpoints

| Method | Path | Description | Input | Response |
|--------|------|-------------|-------|----------|
| GET | `/` | Health check | - | `{"message": "Hello World"}` |
| POST | `/upload_model_pth` | Upload a trained model file | `file` (multipart) | `{"message": "Model uploaded"}` |
| POST | `/predict` | Predict without interpretation | `file` (image/DICOM) | `{"prediction": "NORMAL\|PNEUMONIA", "confidence": 0.95}` |
| POST | `/predict_with_interpretation` | Predict + Gemini interpretation | `file` (image/DICOM) | `{"prediction": "...", "confidence": 0.95, "interpretation": "..."}` |
| POST | `/predict_with_interpretation_groq` | Predict + Groq interpretation | `file` (image/DICOM) | `{"prediction": "...", "confidence": 0.95, "interpretation": "..."}` |
| POST | `/predict_with_interpretation_openai` | Predict + OpenAI interpretation | `file` (image/DICOM) | `{"prediction": "...", "confidence": 0.95, "interpretation": "..."}` |

### Example Request

```bash
# Basic prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@chest_xray.jpg"

# Prediction with Gemini interpretation
curl -X POST http://localhost:8000/predict_with_interpretation \
  -F "file=@chest_xray.dcm"
```

### Example Response

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.92,
  "interpretation": "The AI model has classified this chest X-ray as showing signs of pneumonia with 92% confidence. The findings suggest..."
}
```

## Image Processing

The service handles two image formats:

### Standard Images (JPEG, PNG)
1. Load with PIL
2. Convert to RGB
3. Resize to 224x224
4. Normalize with ImageNet statistics

### DICOM Images (.dcm)
1. Read with pydicom
2. Extract pixel array
3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast
   - clipLimit: 2.0, tileGridSize: (8, 8)
4. Resize to 224x224
5. Convert grayscale to RGB (3-channel stack)
6. Normalize with ImageNet statistics

## CORS Configuration

The API allows requests from these origins:

- `http://localhost:3000`
- `http://localhost:5173`
- `http://127.0.0.1:3000`
- `http://127.0.0.1:5173`

If your frontend runs on a different port, update the `allow_origins` list in `api/api_endpoints.py`.

## Kaggle Integration

For remote model training using Kaggle GPUs:

1. **Generate notebook:** `python scripts/generators/generate_notebook_v3.py`
2. **Push to Kaggle:** `python scripts/send_notebook_to_kaggle.py`
3. **Pull trained model:** `python scripts/pull_trained_model_from_kaggle.py`

The system supports both token-based (`KAGGLE_API_TOKEN`) and legacy (`KAGGLE_USERNAME` + `KAGGLE_KEY`) authentication.

## Project Structure

```
backend-python-ml_integration/
├── api/
│   ├── main.py                 # Server entry point (uvicorn)
│   ├── api_endpoints.py        # FastAPI route definitions + CORS
│   ├── error_handlers.py       # Custom exception handlers
│   ├── dependencies.py         # Shared dependencies
│   ├── models/
│   │   └── prediction.py       # Model loading, prediction, AI interpretation
│   └── utils/
│       └── image_processing.py # Image/DICOM preprocessing
├── model/
│   └── model.py                # ResNet architecture definition
├── train/
│   └── train.py                # Training loop and evaluation
├── preprocess/
│   └── preprocess.py           # Dataset loading and transforms
├── scripts/
│   ├── send_notebook_to_kaggle.py
│   ├── pull_trained_model_from_kaggle.py
│   └── generators/             # Notebook generation scripts
├── output/                     # Trained model files (cnn.pth)
├── logs/                       # Application logs
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

### Model Not Found

```
FileNotFoundError: output/cnn.pth not found
```

Place your trained model at `output/cnn.pth`. See [Model Setup](#model-setup).

### CORS Errors

If the frontend gets CORS errors, add your frontend URL to the `allow_origins` list in `api/api_endpoints.py`.

### API Key Errors

- **Gemini:** Get a key from [Google AI Studio](https://aistudio.google.com/apikey)
- **Groq:** Get a key from [Groq Console](https://console.groq.com/keys)
- **OpenAI:** Get a key from [OpenAI Platform](https://platform.openai.com/api-keys)

Set the keys in your `.env` file.

### GPU vs CPU

The service automatically detects CUDA availability:
- **GPU available:** Model runs on GPU (faster inference)
- **No GPU:** Falls back to CPU (slower but functional)

No configuration needed - it's automatic via `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

### Gemini Model Fallback

The Gemini endpoint tries `gemini-1.5-pro` first. If that model is unavailable, it falls back to `gemini-pro`. If both fail, a default interpretation message is returned.

### Port Already in Use

```bash
# Find and kill the process using port 8000
# Linux/macOS
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

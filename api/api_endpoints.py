import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from api.utils.image_processing import process_image_file as process_image, process_dicom_with_clahe
from api.models.prediction import load_model, classes, generate_interpretation_gemini, generate_interpretation_groq, generate_interpretation_openai
from api.error_handlers import (
    not_found_handler,
    internal_server_error_handler,
    validation_exception_handler
)
import logging
import os
import datetime
import google.generativeai as genai
from dotenv import load_dotenv

# Initialize FastAPI app and load environment variables
app = FastAPI()
load_dotenv()

# Set up logging
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
log_file = os.path.join(logs_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Include custom error handlers
app.add_exception_handler(404, not_found_handler)
app.add_exception_handler(500, internal_server_error_handler)
app.add_exception_handler(422, validation_exception_handler)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("resnet50_model.pth").to(device)

# Set up the Gemini API client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload_model_pth")
async def upload_model_pth(file: UploadFile = File(...)):
    try:
        model_path = os.path.join("output", file.filename)
        with open(model_path, "wb") as buffer:
            buffer.write(file.file.read())
        logger.info(f"Model file '{file.filename}' uploaded successfully")
        return JSONResponse(content={"message": f"Model file '{file.filename}' uploaded successfully"})

    except Exception as e:
        logger.error(f"Model upload failed: {e}")
        raise HTTPException(status_code=500, detail="Model upload failed")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image_tensor = process_image(image_data).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            prediction = classes[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()

        logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
        return JSONResponse(content={"prediction": prediction, "confidence": confidence})

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_with_interpretation")
async def predict_with_interpretation(file: UploadFile = File(...)):
    try:
        # Determine file type based on extension and process accordingly
        filename = file.filename
        image_data = await file.read()

        # Use appropriate processing function based on file type
        if filename.endswith(".dcm"):
            try:
                image_tensor = process_dicom_with_clahe(image_data).to(device)
            except Exception as e:
                logger.error(f"Error processing DICOM image: {e}")
                raise HTTPException(status_code=400, detail="Invalid DICOM file format")
        else:
            try:
                image_tensor = process_image(image_data).to(device)
            except Exception as e:
                logger.error(f"Error processing image file: {e}")
                raise HTTPException(status_code=400, detail="Invalid image file format")

        # Perform the prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            prediction = classes[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()

        # Generate interpretation using Gemini API with the image
        interpretation = generate_interpretation_gemini(prediction, confidence, image_data)

        logger.info(f"Prediction: {prediction}, Confidence: {confidence}, Interpretation: {interpretation}")

        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence,
            "interpretation": interpretation
        })

    except HTTPException as e:
        # FastAPI-specific exception for client errors
        logger.error(f"Prediction with interpretation failed: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Prediction with interpretation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/predict_with_interpretation_groq")
async def predict_with_interpretation_groq(file: UploadFile = File(...)):
    try:
        # Determine file type based on extension and process accordingly
        filename = file.filename
        image_data = await file.read()

        # Use appropriate processing function based on file type
        if filename.endswith(".dcm"):
            try:
                image_tensor = process_dicom_with_clahe(image_data).to(device)
            except Exception as e:
                logger.error(f"Error processing DICOM image: {e}")
                raise HTTPException(status_code=400, detail="Invalid DICOM file format")
        else:
            try:
                image_tensor = process_image(image_data).to(device)
            except Exception as e:
                logger.error(f"Error processing image file: {e}")
                raise HTTPException(status_code=400, detail="Invalid image file format")

        # Perform the prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            prediction = classes[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()

        # Generate interpretation using Gemini API with the image
        interpretation = generate_interpretation_groq(prediction, confidence, image_data)

        logger.info(f"Prediction: {prediction}, Confidence: {confidence}, Interpretation: {interpretation}")

        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence,
            "interpretation": interpretation
        })

    except HTTPException as e:
        # FastAPI-specific exception for client errors
        logger.error(f"Prediction with interpretation failed: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Prediction with interpretation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@app.post("/predict_with_interpretation_openai")
async def predict_with_interpretation_openai(file: UploadFile = File(...)):
    try:
        # Determine file type based on extension and process accordingly
        filename = file.filename
        image_data = await file.read()

        # Use appropriate processing function based on file type
        if filename.endswith(".dcm"):
            try:
                image_tensor = process_dicom_with_clahe(image_data).to(device)
            except Exception as e:
                logger.error(f"Error processing DICOM image: {e}")
                raise HTTPException(status_code=400, detail="Invalid DICOM file format")
        else:
            try:
                image_tensor = process_image_file(image_data).to(device)
            except Exception as e:
                logger.error(f"Error processing image file: {e}")
                raise HTTPException(status_code=400, detail="Invalid image file format")

        # Perform the prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            prediction = classes[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()

        # Generate interpretation using OpenAI API with the image
        interpretation = generate_interpretation_openai(prediction, confidence, image_data)

        logger.info(f"Prediction: {prediction}, Confidence: {confidence}, Interpretation: {interpretation}")

        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence,
            "interpretation": interpretation
        })

    except HTTPException as e:
        logger.error(f"Prediction with interpretation failed: {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Prediction with interpretation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

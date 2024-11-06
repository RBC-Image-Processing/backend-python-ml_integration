# import base64
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.responses import JSONResponse
# import torch
# import torch.nn.functional as F
# from api.utils.image_processing import process_image_file as process_image
# from api.models.prediction import load_model, classes
# from api.error_handlers import (
#     not_found_handler,
#     internal_server_error_handler,
#     validation_exception_handler
# )
# import logging
# import os
# import datetime
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Initialize FastAPI app and load environment variables
# app = FastAPI()
# load_dotenv()

# # Set up logging
# logs_dir = os.path.join(os.getcwd(), "logs")
# os.makedirs(logs_dir, exist_ok=True)
# log_file = os.path.join(logs_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
# logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Include custom error handlers
# app.add_exception_handler(404, not_found_handler)
# app.add_exception_handler(500, internal_server_error_handler)
# app.add_exception_handler(422, validation_exception_handler)

# # Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = load_model("resnet50_model.pth").to(device)

# # Set up the Gemini API client
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image_data = await file.read()
#         image_tensor = process_image(image_data).to(device)

#         with torch.no_grad():
#             output = model(image_tensor)
#             probabilities = F.softmax(output, dim=1)
#             predicted = torch.argmax(probabilities, dim=1)
#             prediction = classes[predicted.item()]
#             confidence = probabilities[0][predicted.item()].item()

#         logger.info(f"Prediction: {prediction}, Confidence: {confidence}")
#         return JSONResponse(content={"prediction": prediction, "confidence": confidence})

#     except Exception as e:
#         logger.error(f"Prediction failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict_with_interpretation")
# async def predict_with_interpretation(file: UploadFile = File(...)):
#     try:
#         image_data = await file.read()
#         image_tensor = process_image(image_data).to(device)

#         with torch.no_grad():
#             output = model(image_tensor)
#             probabilities = F.softmax(output, dim=1)
#             predicted = torch.argmax(probabilities, dim=1)
#             prediction = classes[predicted.item()]
#             confidence = probabilities[0][predicted.item()].item()

#         # Generate interpretation using Gemini API with the image
#         interpretation = generate_interpretation_gemini(prediction, confidence, image_data)

#         logger.info(f"Prediction: {prediction}, Confidence: {confidence}, Interpretation: {interpretation}")

#         return JSONResponse(content={
#             "prediction": prediction,
#             "confidence": confidence,
#             "interpretation": interpretation
#         })

#     except Exception as e:
#         logger.error(f"Prediction with interpretation failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# def generate_interpretation_gemini(prediction: str, confidence: float, image_data: bytes) -> str:
#     """Generate an interpretation using Gemini API based on the prediction, confidence, and image data."""
#     # Encode image data as base64
#     image_base64 = base64.b64encode(image_data).decode('utf-8')

#     prompt = (
#         f"The AI model has analyzed the provided medical image and classified it as '{prediction}' with a confidence score of {confidence:.2f}. "
#         f"Please provide a thorough clinical interpretation that explains the possible implications of this finding. "
#         f"The interpretation should consider the likely clinical scenarios that align with a '{prediction}' diagnosis, along with possible symptoms, "
#         f"treatment considerations, and advice for the next steps a patient might take. Additionally, the image data is attached as encoded text "
#         f"for any relevant context: {image_base64}. "
#         f"Your interpretation should be clear, patient-centered, and should emphasize any limitations of the AI model's confidence score in this context."
#     )

#     logger.info(f"Generating interpretation with Gemini API for prediction: {prediction}, confidence: {confidence}")

#     try:
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(prompt)

#         interpretation = response.text
#         logger.info(f"Generated interpretation: {interpretation}")
#         return interpretation

#     except Exception as e:
#         logger.error(f"Failed to get response from Gemini API: {e}")
#         return "An error occurred while generating the interpretation. Please try again later."

import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from api.utils.image_processing import process_image_file, process_dicom_with_clahe
from api.models.prediction import load_model, classes, generate_interpretation_gemini
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

# @app.post("/predict_with_interpretation")
# async def predict_with_interpretation(file: UploadFile = File(...)):
#     try:
#         image_data = await file.read()
#         image_tensor = process_image(image_data).to(device)

#         with torch.no_grad():
#             output = model(image_tensor)
#             probabilities = F.softmax(output, dim=1)
#             predicted = torch.argmax(probabilities, dim=1)
#             prediction = classes[predicted.item()]
#             confidence = probabilities[0][predicted.item()].item()

#         # Generate interpretation using Gemini API with the image
#         interpretation = generate_interpretation_gemini(prediction, confidence, image_data)

#         logger.info(f"Prediction: {prediction}, Confidence: {confidence}, Interpretation: {interpretation}")

#         return JSONResponse(content={
#             "prediction": prediction,
#             "confidence": confidence,
#             "interpretation": interpretation
#         })

#     except Exception as e:
#         logger.error(f"Prediction with interpretation failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict_with_interpretation")
async def predict_with_interpretation(file: UploadFile = File(...)):
    try:
        # Determine file type based on extension and process accordingly
        filename = file.filename
        image_data = await file.read()
        
        # Use appropriate processing function based on file type
        if filename.endswith(".dcm"):
            image_tensor = process_dicom_with_clahe(image_data).to(device)
        else:
            image_tensor = process_image_file(image_data).to(device)

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

    except Exception as e:
        logger.error(f"Prediction with interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
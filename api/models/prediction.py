import base64
import os
import torch
import torch.nn.functional as F
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import models
from api.utils.image_processing import process_dicom_with_clahe
import requests
import google.generativeai as genai 


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define model directory and logging
MODEL_DIR = os.path.abspath("output")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# Load classes
classes = ["NORMAL", "PNEUMONIA"]

def load_model(model_name: str = "resnet50_model.pth"):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_name}' not found in {MODEL_DIR}. Ensure the correct path.")
        return None

    try:
        # Initialize the model architecture
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

        # Load the state_dict (weights only)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logger.info(f"Model '{model_name}' successfully loaded on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        return None

def predict(model, dicom_data: bytes):
    if model is None:
        return {"error": "Model loading failed. Cannot perform prediction."}

    # Preprocess the DICOM data
    image_tensor = process_dicom_with_clahe(dicom_data).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted = torch.max(probabilities, 1)[1]
        prediction = classes[predicted.item()]
        confidence = probabilities[0][predicted].item()

    return {"prediction": prediction, "confidence": confidence}

# Function to call Gemini API
def getPromptGemini(prompt: str) -> str:
    """Calls Gemini API with the prompt and retrieves the generated interpretation."""
    api_key = os.getenv("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "prompt": prompt,
        "maxTokens": 100
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        response_data = response.json()
        return response_data['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        logger.error(f"Failed to get response from Gemini API: {e}")
        return "Error generating interpretation from Gemini API."

# def generate_interpretation_gemini(prediction: str, confidence: float) -> str:
#     """Generate a Gemini API-based interpretation based on the prediction and confidence."""
#     prompt = (
#         f"The medical image was analyzed, and the results indicate a classification of '{prediction}' with a confidence score of {confidence:.2f}. "
#         f"Please provide a detailed medical interpretation explaining what this means for the patient's health."
#     )
    
#     logger.info(f"Generating interpretation with Gemini API for prediction: {prediction}, confidence: {confidence}")
#     interpretation = getPromptGemini(prompt)
#     logger.info(f"Generated interpretation: {interpretation}")

#     return interpretation

def generate_interpretation_gemini(prediction: str, confidence: float, image_data: bytes) -> str:
    """Generate an interpretation using Gemini API based on the prediction, confidence, and image data."""
    # Encode image data as base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    prompt = (
        f"The AI model has analyzed the provided medical image and classified it as '{prediction}' with a confidence score of {confidence:.2f}. "
        f"Please provide a thorough clinical interpretation that explains the possible implications of this finding. "
        f"The interpretation should consider the likely clinical scenarios that align with a '{prediction}' diagnosis, along with possible symptoms, "
        f"treatment considerations, and advice for the next steps a patient might take. Additionally, the image data is attached as encoded text "
        f"for any relevant context: {image_base64}. "
        f"Your interpretation should be clear, patient-centered, and should emphasize any limitations of the AI model's confidence score in this context."
    )

    logger.info(f"Generating interpretation with Gemini API for prediction: {prediction}, confidence: {confidence}")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        interpretation = response.text
        logger.info(f"Generated interpretation: {interpretation}")
        return interpretation

    except Exception as e:
        logger.error(f"Failed to get response from Gemini API: {e}")
        return "An error occurred while generating the interpretation. Please try again later."
import openai
import os
import base64
import torch
import torch.nn.functional as F
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import models
from api.utils.image_processing import process_dicom_with_clahe
import requests
import google.generativeai as genai
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

# Verify the API keys
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))
print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))

# Initialize Groq client with API key
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

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
        predicted = torch.argmax(probabilities, dim=1)
        prediction = classes[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()

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

def generate_interpretation_groq(prediction: str, confidence: float, image_data: bytes) -> str:
    """Generate an interpretation using Groq API based on prediction, confidence, and image data, while fitting the token limit."""

    # Encode image data as base64 to include in the prompt, limiting the image data size
    image_base64 = base64.b64encode(image_data[:3000]).decode('utf-8')  # Use only the first 3000 bytes for brevity

    # Define a concise prompt that fits the token limit
    prompt = (
        f"The model classified a medical image as '{prediction}' with confidence {confidence:.2f}. "
        f"Provide an interpretation, detailing possible scenarios, symptoms, and next steps for a '{prediction}' diagnosis. "
        f"Consider the model’s confidence score and limitations. Encoded image data (shortened): {image_base64}."
    )

    # Log the prompt for debugging
    logger.info(f"Sending prompt to Groq API for prediction: {prediction}, confidence: {confidence}")

    try:
        # Make the API call to Groq for chat completion
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a clinical assistant providing concise interpretations for medical AI results."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192"  # Specify the model
        )

        # Extract and return the generated interpretation from the response
        interpretation = chat_completion.choices[0].message.content
        logger.info(f"Generated interpretation from Groq: {interpretation}")
        return interpretation

    except Exception as e:
        # Log any errors and return an error message
        logger.error(f"Failed to get response from Groq API: {e}")
        return "An error occurred while generating the interpretation with Groq. Please try again later."


def check_available_models():
    """Check and log available models in OpenAI."""
    try:
        models = openai.Model.list()
        available_models = [model.id for model in models['data']]
        logging.info(f"Available OpenAI models: {available_models}")
    except Exception as e:
        logging.error(f"Failed to retrieve available OpenAI models: {e}")

# Call this function to log available models at startup
check_available_models()

def generate_interpretation_openai(prediction: str, confidence: float, image_data: bytes) -> str:
    """Generate an interpretation using OpenAI API based on prediction, confidence, and image data."""
    # Initialize the OpenAI client
    client = OpenAI(
        api_key=os.getenv('OPEN_AI_API_KEY') 
        )  # Ensure OPENAI_API_KEY is set in environment variables
    
    # Convert confidence to percentage for clearer communication
    confidence_percentage = confidence * 100

    try:
        # Create a message content list with the medical context and findings
        messages = [
            {
                "role": "system",
                "content": "You are a medical AI assistant specializing in interpreting medical image classifications. Provide detailed clinical interpretations while acknowledging any uncertainties or limitations."
            },
            {
                "role": "user",
                "content": (
                    f"Based on the medical image analysis, the following classification was made:\n"
                    f"- Diagnosis: {prediction}\n"
                    f"- Confidence Level: {confidence_percentage:.1f}%\n\n"
                    f"Please provide:\n"
                    f"1. Clinical interpretation\n"
                    f"2. Likely symptoms\n"
                    f"3. Recommended next steps\n"
                    f"4. Any relevant limitations based on the confidence score"
                )
            }
        ]

        logging.info(f"Sending request to OpenAI API - Prediction: {prediction}, Confidence: {confidence_percentage:.1f}%")
        
        # Make the API call using the chat completions endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Changed from gpt-4 to gpt-3.5-turbo
            messages=messages,
            temperature=0.4,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        # Extract the interpretation from the response
        interpretation = response.choices[0].message.content.strip()
        
        logging.info("Successfully generated interpretation from OpenAI")
        return interpretation

    except Exception as e:
        error_msg = f"OpenAI API error: {str(e)}"
        logging.error(error_msg)
        return (
            "An error occurred while generating the medical interpretation. "
            "Please consult with a healthcare professional for accurate diagnosis and treatment options."
        )

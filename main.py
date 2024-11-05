# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# import torch
# import torch.nn.functional as F
# from model.model import process_dicom_with_clahe  # Import the preprocessing function
# from model.model import initiate_model

# app = FastAPI()


# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


# #path to the model
# VIT_MODEL_PATH = "/Users/brightsmac/PycharmProjects/RBC_BackEnd/model/Vit.pth"
# CNN_MODEL_PATH = "/Users/brightsmac/PycharmProjects/RBC_BackEnd/model/cnn_impr.pth"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # def load_model(model_type, path):
# #     if model_type == 'Vit':
# #         model = initiate_model(model_type, path)
# #     else:
# #         #  Load the model
# #         model = initiate_model('resnet18')
# #         model.load_state_dict(torch.load(path, map_location=device))
# #
# #     model.to(device)
# #     model.eval()

# def load_model(path):
#     try:
#         # Load the model
#         model = torch.load(path, map_location=device)

#         # Move the model to the appropriate device (GPU/CPU)
#         model.to(device)

#         # Set the model to evaluation mode
#         model.eval()

#         # Print/log a success message
#         print(f"Model successfully loaded from {path} and moved to {device}")

#         return model

#     except Exception as e:
#         # Handle any errors during loading
#         print(f"Failed to load the model: {str(e)}")
#         return None


# classes = ['NORMAL', "PNEUMONIA"]




# # load the model
# model = load_model(CNN_MODEL_PATH)

# # model = initiate_model("resnet18")
# # model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
# # model.to(device)
# # model.eval()


# @app.get("/predict")
# async def predict(model_type: str, file: UploadFile = File(...)):
#     try:
#         #read the uploaded dicom data
#         dicom_data = await file.read()

#         #preprocess the DICOM image using the imported function
#         image_tensor = process_dicom_with_clahe(dicom_data).to(device)


#         #perform the prediction
#         with torch.no_grad():
#             output = model(image_tensor)
#             if model_type == 'Vit':
#                 probabilities = F.softmax(output.logits, dim=1)  # Get probabilities
#             else:
#                 probabilities = F.softmax(output, dim=1)
#             predicted = torch.max(probabilities, 1)[1]  # Get predicted class
#             prediction = classes[predicted.item()]
#             confidence = probabilities[0][predicted].item()  # Confidence score

#             # Return the result as a JSON response
#         return JSONResponse(content={"prediction": prediction, "confidence": confidence})


#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)

# # READ THE PAPER :: file:///Users/brightsmac/Desktop/SEM_FILES/LAST%20FALL/PRACTICUM/papers/1728-6521-1-PB.pdf

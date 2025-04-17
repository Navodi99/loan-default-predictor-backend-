from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
from tensorflow.keras.models import load_model


# Load preprocessing objects
with open('model/preprocessing.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

scaler = preprocessing['scaler']
encoder = preprocessing['encoder']
num_cols = preprocessing['num_cols']
cat_cols = preprocessing['cat_cols']
binary_cols = preprocessing['binary_cols']

# Load the trained model
model = load_model('model/loan_default_model.keras')

# Initialize FastAPI app
app = FastAPI()

# Initialize CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request model using Pydantic
class LoanApplication(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    Education: str
    EmploymentType: str
    MaritalStatus: str
    LoanPurpose: str
    HasMortgage: str
    HasDependents: str
    HasCoSigner: str

# Preprocessing and Prediction Function
def predict_default(input_data):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data.dict()])

    # Convert binary values ('Yes'/'No') to 1/0
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0}).astype(int)

    # One-hot encode categorical data
    encoded_data = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))

    # Combine numerical and encoded categorical data
    final_input = pd.concat([input_df[num_cols], encoded_df], axis=1)

    # Scale input data
    final_input_scaled = scaler.transform(final_input)

    # Make prediction
    prediction_prob = float(model.predict(final_input_scaled)[0][0]) 
    prediction_class = int(prediction_prob > 0.5)

    return {
        'prediction_probability': round(prediction_prob, 2),
        'prediction_class': 'Default' if prediction_class == 1 else 'Non-Default'
    }

# API Endpoint
@app.post('/api/predict')
def predict(application: LoanApplication):
    print(application);
    result = predict_default(application)
    return {
        'status': 'success',
        'prediction': result
    }

# Root endpoint
@app.get('/')
def root():
    return {"message": "Loan Default Prediction API is running!"}
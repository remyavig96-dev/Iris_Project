from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the saved model
model = joblib.load("iris_model.pkl")

flowers = ["Setosa", "Versicolor", "Virginica"]

@app.get("/")
def home():
    return {"message": "Iris Prediction API"}

@app.get("/predict")
def predict():

    result = model.predict([[5.1, 3.5, 1.4, 0.2]])

    return {
        "prediction": flowers[result[0]]
    }
import joblib

model = joblib.load("iris_model.pkl")

prediction = model.predict([[5.1,3.5,1.4,0.2]])

print(prediction)

flowers=[
    "Setosa",
    "Versicolor",
    "Virginica"
]

print(flowers[prediction[0]])
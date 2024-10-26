# iris_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the Iris dataset and train a simple model
iris = load_iris()
model = LogisticRegression(max_iter=200)
model.fit(iris.data, iris.target)

# Define the FastAPI app
app = FastAPI()


# Define input data structure
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Define the target names based on the dataset
target_names = iris.target_names


@app.post("/predict")
def predict(iris: IrisRequest):
    # Prepare the input data
    data = np.array([
        iris.sepal_length,
        iris.sepal_width,
        iris.petal_length,
        iris.petal_width
    ]).reshape(1, -1)

    # Predict the class
    prediction = model.predict(data)
    predicted_class = target_names[prediction[0]]  # Access `target_names` directly
    return {"prediction": predicted_class}


@app.get("/")
def hello():
    return {"Hello": "World"}
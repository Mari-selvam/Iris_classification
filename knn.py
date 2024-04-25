from joblib import load
from sklearn.neighbors import KNeighborsClassifier

model = load('model.joblib')

def prediction(value: list):
    y_pred = model.predict([value])
    return y_pred



from joblib import load
import numpy as np

model = load("model.joblib")
arr = np.array([12, 15.5, 4, 8, 512, 4, 1.98, 1])
pred = model.predict([arr])
print((f"{pred[0]} Euros." ))
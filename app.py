import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


X = np.array([[i] for i in range(1, 11)])  # Entrada
y = np.array([i+1 for i in range(1, 11)])  # Salida esperada


model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(1)  # Predicción
])

model.compile(optimizer='adam', loss='mse')


model.fit(X, y, epochs=200, verbose=0)


test_input = np.array([[11], [12], [13]])
predictions = model.predict(test_input)


for i, pred in zip(test_input, predictions):
    print(f"Entrada: {i[0]}, Predicción: {pred[0]:.2f}")

y_pred = model.predict(X)
plt.plot(X, y, label="Real")
plt.plot(X, y_pred, label="Predicción", linestyle='--')
plt.legend()
plt.title("Predicción del siguiente número")
plt.xlabel("Número")
plt.ylabel("Siguiente número")
plt.show()

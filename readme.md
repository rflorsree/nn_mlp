# Predicción del siguiente número con MLP

Ejemplo **básico y práctico** de una red neuronal (Perceptrón Multicapa) que aprende a predecir el **siguiente número en una secuencia** segun el dataset proporcionado no es dependiente de la ejecucion.

---

## Requisitos

- Python 3.7+
- TensorFlow
- NumPy
- Matplotlib
- Pandas (opcional)

---

## Justificación

Este modelo está diseñado para trabajar con **datos univariantes estáticos**, es decir, entradas que **no son secuencias** sino valores individuales por muestra. Por ejemplo, entradas como `[x]`, donde cada `x` representa una sola característica numérica, sin contexto temporal.

## Arquitectura del modelo

```python
model = Sequential([
    Dense(8, activation='relu', input_shape=(1,)),
    Dense(1)
])
```
## Capas ¿

### Capa Densa (oculta)

- **Unidades**: 8 neuronas.
- **Función de activación**: `ReLU`, que permite al modelo aprender patrones no lineales.
- **Entrada esperada**: un valor escalar por muestra (`input_shape=(1,)`).

### Capa Densa (salida)

- **Unidades**: 1 neurona.
- **Función de activación**: lineal (por defecto).
- **Propósito**: Devuelve un único valor como salida final. Puede utilizarse tanto para tareas de **regresión** (predicción de valores continuos) como de **clasificación binaria** 


## Referencias

- [Dense layer - TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)  
- Sotelo, J. A. L. (2023, 68). *Deep Learning: teoría y aplicaciones*. Marcombo.

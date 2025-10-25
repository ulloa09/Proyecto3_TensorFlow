import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def build_mlp_model(input_shape, params):
    """Construye un modelo MLP con ReLU"""
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    dense_layers = params.get("dense_layers", 2)
    units = params.get("dense_units", 128)
    activation = 'relu' # Usando ReLU
    
    for _ in range(dense_layers):
        model.add(Dense(units, activation=activation))
        
    model.add(Dense(3, activation='softmax')) # 3 clases (0, 1, 2)
    
    optimizer = "adam"
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
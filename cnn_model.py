import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input

def build_cnn_model(input_shape, params):
    """Construye un modelo CNN con Tanh (sin Dropout)"""
    model = Sequential()
    model.add(Input(shape=(input_shape, 1)))
    
    num_filters = params.get("conv_filters", 32)
    conv_layers = params.get("conv_layers", 2)
    activation = 'tanh' # Usando Tanh
    
    for _ in range(conv_layers):
        model.add(Conv1D(filters=num_filters, kernel_size=3, activation=activation, padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        num_filters *= 2
        
    dense_units = params.get("dense_units", 64)
    model.add(Flatten())
    model.add(Dense(dense_units, activation=activation))
    
    model.add(Dense(3, activation='softmax')) # 3 clases (0, 1, 2)
    
    # Mantenemos el clipvalue, que es buena práctica
    optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
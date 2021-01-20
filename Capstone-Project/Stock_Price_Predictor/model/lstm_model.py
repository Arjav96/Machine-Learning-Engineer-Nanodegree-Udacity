from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM,Dropout
import matplotlib.pyplot as plt

def build_simple_model(X_train, y_train, X_val, y_val):
    """ Build single-layered LSTM RNN model."""
    model = Sequential()

    model.add(LSTM(32, return_sequences=False, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mae', optimizer='adam', metrics=['mean_squared_error'])
    
    history = model.fit(X_train,
                        y_train, 
                        epochs = 20,
                        validation_data=(X_val,y_val),
                        batch_size = 32)
    
    """Displays model summary."""
    model.summary()
    
    return model, history

def build_improved_model(X_train, y_train, X_val, y_val):
    """ Build multi-layered LSTM RNN improved model using appropriate dropouts."""
    model = Sequential()
    # First layer LSTM
    model.add(LSTM(128, return_sequences=True, input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    # Second layer LSTM
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mae', optimizer='adam', metrics=['mean_squared_error'])
    
    history = model.fit(X_train,
                        y_train, 
                        epochs = 20,
                        validation_data=(X_val,y_val),
                        batch_size = 32)
    
    """Displays model summary."""
    model.summary()
    
    return model, history
    
def plot_loss(history):
    """Plots model history curve to analyze training_loss vs validation loss."""
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.xlabel('No of epochs')
    plt.ylabel('Loss (mean absolute error)')
    plt.legend()
    plt.show()
    
    
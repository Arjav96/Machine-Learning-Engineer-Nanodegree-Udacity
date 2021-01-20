import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

def predict_and_plot(model, X, y_actual):
    """ Perform prediction on features X then displays r2_score and mean_squared_error 
    and later on displays the prediction vs actual curve.""" 
    
    # Predict
    y_pred = model.predict(X)
    
    # Displays r2_score
    print("r2_score: {}".format(r2_score(y_actual, y_pred)))
    # Displays mean_squared_error
    print("mean_squared_error: {}".format(mean_squared_error(y_actual, y_pred)))
    # Displays root_mean_squared_error
    print("root_mean_squared_error: {}".format(np.sqrt(mean_squared_error(y_actual, y_pred))))
   
    # Plot prediction vs actual
    plt.plot(y_pred, label='prediction')
    plt.plot(y_actual, label='actual')
    plt.xlabel('Trading days')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()
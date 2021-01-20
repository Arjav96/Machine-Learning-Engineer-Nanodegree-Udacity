from sklearn.linear_model import LinearRegression

"""Benchmark model."""

def build_benchmark_model(X, y):
    """ Building Linear Regression model as the benchmark model."""
    model = LinearRegression()
    model = model.fit(X, y)
    
    """Displays model summary."""
    print("Model coefficient: {}".format(model.coef_))
    print("Model intercept: {}".format(model.intercept_))
    
    return model

from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, train_series):
        """
        Fit ARIMA model to training data.
        :param train_series: 1D array-like time series
        """
        self.model = ARIMA(train_series, order=self.order)
        self.fitted_model = self.model.fit()

    def forecast(self, steps):
        """
        Forecast future values.
        :param steps: number of future steps to forecast
        :return: forecasted values (numpy array)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)

    def summary(self):
        """
        Print the model summary.
        """
        if self.fitted_model is None:
            print("Model is not fitted yet.")
        else:
            print(self.fitted_model.summary())

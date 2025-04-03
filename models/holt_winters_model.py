# models/holt_winters_model.py

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np

class HoltWintersModel:
    def __init__(self, seasonal='add', seasonal_periods=12, trend='add'):
        """
        Initialize the Holt-Winters model.
        
        :param seasonal: 'add' or 'mul' for additive or multiplicative seasonality
        :param seasonal_periods: Number of periods in a season (e.g., 12 for monthly data)
        :param trend: 'add', 'mul', or None
        """
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.model = None
        self.fitted_model = None

    def fit(self, train_series):
        """
        Fit the model to training data.
        
        :param train_series: 1D array-like time series
        """
        self.model = ExponentialSmoothing(
            train_series,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            trend=self.trend
        )
        self.fitted_model = self.model.fit()

    def forecast(self, steps):
        """
        Forecast future values.
        
        :param steps: number of future steps to forecast
        :return: forecasted values (numpy array)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting.")
        forecast = self.fitted_model.forecast(steps)
        return np.array(forecast)

    def summary(self):
        """
        Print model parameters and fit information.
        """
        if self.fitted_model is None:
            print("Model is not fitted yet.")
        else:
            print("Model params:", self.fitted_model.params)

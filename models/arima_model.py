import numpy as np
import pmdarima as pm

class ARIMAModel:
    def __init__(
        self,
        start_p=1,  # initial guess for p
        start_q=1,  # initial guess for q
        max_p=5,    # maximum p
        max_q=5,    # maximum q
        seasonal=False,
        m=1,        # season length, e.g. 7 for weekly in daily data
        d=None,     # if None, auto-arima tries to find it
        D=None,     # if None, auto-arima tries to find it
        trace=False,  # if True, prints debugging info
        **auto_arima_kwargs  # any extra parameters you want to pass
    ):
        """
        Initialize the Auto ARIMA model with typical parameters.
        Customize them based on your needs.
        """
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self.m = m
        self.d = d
        self.D = D
        self.trace = trace
        self.auto_arima_kwargs = auto_arima_kwargs

        # This will hold the fitted auto_arima model
        self.model = None

    def fit(self, train_series):
        """
        Fit auto-ARIMA to training data.
        
        :param train_series: 1D array-like time series (pandas Series or NumPy array).
        """
        self.model = pm.auto_arima(
            train_series,
            start_p=self.start_p,
            start_q=self.start_q,
            max_p=self.max_p,
            max_q=self.max_q,
            seasonal=self.seasonal,
            m=self.m,
            d=self.d,
            D=self.D,
            trace=self.trace,
            error_action='ignore',      # ignore non-invertible cases
            suppress_warnings=True,     # don't want convergence warnings
            stepwise=True,             # apply stepwise algorithm to speed-up
            **self.auto_arima_kwargs
        )

    def forecast(self, steps):
        """
        Forecast future values.
        
        :param steps: number of future steps to forecast
        :return: forecasted values (numpy array)
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting.")
        # auto_arima uses .predict(n_periods=...)
        forecast_vals = self.model.predict(n_periods=steps)
        return np.array(forecast_vals)

    def summary(self):
        """
        Print the model summary (if available).
        """
        if self.model is None:
            print("Model is not fitted yet.")
        else:
            print(self.model.summary())

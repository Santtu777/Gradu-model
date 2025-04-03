# models/xgboost_model.py

import xgboost as xgb
import numpy as np

class XGBoostModel:
    def __init__(self, **kwargs):
        """
        Initialize an XGBoost regressor with optional hyperparameters.
        
        For example:
        XGBoostModel(n_estimators=100, max_depth=3, learning_rate=0.1, ...)
        
        :param kwargs: XGBoost hyperparameters (n_estimators, max_depth, etc.)
        """
        # Store parameters (useful if you want to reference them later)
        self.params = kwargs
        
        # Create the XGBoost regressor
        self.model = xgb.XGBRegressor(**kwargs)

    def fit(self, X_train, y_train, eval_set=None, early_stopping_rounds=None):
        """
        Fit the XGBoost model.
        
        :param X_train: Training features, shape (num_samples, num_features)
        :param y_train: Training labels/targets, shape (num_samples,)
        :param eval_set: A list of (X_val, y_val) pairs for validation
        :param early_stopping_rounds: Enables early stopping if eval metric 
                                      is not improving
        :return: self
        """
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=eval_set, 
            early_stopping_rounds=early_stopping_rounds,
            verbose=(eval_set is not None)
        )
        return self

    def predict(self, X):
        """
        Predict using the trained model.
        
        :param X: Features to predict on, shape (num_samples, num_features)
        :return: Predicted values as a numpy array
        """
        return self.model.predict(X)

    def summary(self):
        """
        Print or return model summary (e.g., feature importances).
        """
        # You can customize what you want to print or return.
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type='gain')
        
        print("XGBoost Feature Importances (by gain):")
        for feature, score in importance.items():
            print(f"  {feature}: {score}")
        
        # If you want a more official summary, you can do:
        # print(self.model)
        # or 
        # print(booster.stats())

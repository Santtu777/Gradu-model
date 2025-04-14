import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model


class LSTMModel:
    def __init__(self, input_shape, hidden_dim, output_dim):
        """
        Initialize a simple LSTM model in TensorFlow/Keras.

        :param input_shape: Shape of a single training example 
                           (e.g., (timesteps, num_features))
        :param hidden_dim: Number of units in the LSTM layer
        :param output_dim: Number of output neurons (e.g., 1 for univariate forecast)
        """
        
        # Define the inputs
        inputs = Input(shape=input_shape)
        
        # Pass through an LSTM layer
        x = LSTM(hidden_dim, activation='tanh')(inputs)
        
        # Final dense layer for output
        outputs = Dense(output_dim, activation='linear')(x)
        
        # Build the Keras Model
        self.model = Model(inputs, outputs)
        
        # Compile with Adam optimizer and MSE loss (typical for regression tasks)
        self.model.compile(optimizer='adam', loss='mse')
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None):
        """
        Train the LSTM model.

        :param X_train: Training data features of shape 
                        (num_samples, timesteps, num_features)
        :param y_train: Training data labels/targets
        :param epochs: Number of epochs to train
        :param batch_size: Batch size for training
        :param validation_data: (X_val, y_val) if you have a validation set
        :return: Training history
        """
        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """
        Generate predictions.

        :param X: Data to predict on, shape (num_samples, timesteps, num_features)
        :return: Model predictions (numpy array)
        """
        return self.model.predict(X)
    
    def summary(self):
        """
        Print the model architecture.
        """
        return self.model.summary()

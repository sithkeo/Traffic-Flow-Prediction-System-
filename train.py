# train.py

"""
Handles model training logic in a reusable, modular format.
Used by main.py or external scripts.
"""

import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model(model_module, X_train, y_train, X_test, y_test, scaler,
                batch_size=32, epochs=50):
    """
    Trains the model from a given module and evaluates performance.

    Args:
        model_module (module): A model script with a build_model(input_shape) function
        X_train, y_train: Training data
        X_test, y_test: Testing data
        scaler: Fitted MinMaxScaler to inverse transform predictions
        batch_size (int): Training batch size
        epochs (int): Number of epochs to train

    Returns:
        dict with predictions, actuals, MAE, RMSE, model object
    """
    model_name = model_module.__name__.split('.')[-1].replace('_model', '').upper()
    print(f"\n--- Training {model_name} ---")

    is_sequence = len(X_train.shape) == 3
    input_shape = X_train.shape[1:] if is_sequence else (X_train.shape[1],)
    X_train_model = X_train if is_sequence else X_train.reshape(X_train.shape[0], -1)
    X_test_model = X_test if is_sequence else X_test.reshape(X_test.shape[0], -1)

    model = model_module.build_model(input_shape)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(X_train_model, y_train,
              validation_split=0.1,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[early_stop],
              verbose=0)

    preds = model.predict(X_test_model)
    preds_rescaled = scaler.inverse_transform(preds)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(y_test_rescaled, preds_rescaled)
    rmse = mean_squared_error(y_test_rescaled, preds_rescaled) ** 0.5

    avg_volume = np.mean(y_test_rescaled)
    mae_percent = (mae / avg_volume) * 100
    print(f"{model_name} MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAE%: {mae_percent:.2f}% of avg. volume ({avg_volume:.2f})")

    return {
        "model": model,
        "predictions": preds_rescaled,
        "actuals": y_test_rescaled,
        "mae": mae,
        "rmse": rmse
    }

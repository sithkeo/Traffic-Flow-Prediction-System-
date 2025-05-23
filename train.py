# train.py

# Training Strategy Notes:
# - We use 90% of training data for actual training and 10% for validation.
# - EarlyStopping monitors this validation loss to decide when to stop.
# - This helps avoid wasted computation and results in a more generalisable model.
# - Training can still run up to `epochs` if improvements continue.

"""
Handles model training logic in a reusable, modular format.
Used by main.py or external scripts.
"""

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
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

    # Determine input shape depending on data type (sequence or flat)
    is_sequence = len(X_train.shape) == 3
    input_shape = X_train.shape[1:] if is_sequence else (X_train.shape[1],)
    X_train_model = X_train if is_sequence else X_train.reshape(X_train.shape[0], -1)
    X_test_model = X_test if is_sequence else X_test.reshape(X_test.shape[0], -1)

    # Build and compile model
    model = model_module.build_model(input_shape)

    # NOTE: EarlyStopping stops training once validation loss stops improving
    early_stop = EarlyStopping(
        monitor="val_loss",         # Track validation loss, not training loss
        patience=5,                 # Wait 5 epochs before stopping
        restore_best_weights=True   # Revert to best-performing weights
    )

    # Train model with 10% of training data used for validation
    history = model.fit(
        X_train_model, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1 # Set to 0 for no output, 1 for progress bar, 2 for one line per epoch
    )

    if history is None:
        raise RuntimeError("[ERROR] model.fit() did not return a history object.")
    else:
        print("[DEBUG] Training history captured successfully.")

    # Save trained model
    model.save(f"output/{model_name}_trained.keras")
    print(f"[INFO] Model saved to output/{model_name}_trained.keras")

    # Predict and rescale outputs
    preds = model.predict(X_test_model)
    preds_rescaled = scaler.inverse_transform(preds)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Evaluate performance
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
        "rmse": rmse,
        "history": history
    }


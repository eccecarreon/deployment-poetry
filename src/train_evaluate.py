import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Function to evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test):

    # Create predictions on the training set
    train_preds = np.rint(model.predict(X_train)) # rint round to the nearest integer
    test_preds = np.rint(model.predict(X_test))

    # Classification report
    train_report = classification_report(y_train, train_preds)
    test_report = classification_report(y_test, test_preds)

    # Confusion matrix
    cm_train = confusion_matrix(y_train, train_preds)
    cm_test = confusion_matrix(y_test, test_preds)

    # Format figures in dark mode
    plt.style.use('dark_background')

    # Plot confusion matrix side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # First axis
    axes[0].text(0.01, 0.05, str(train_report), font_properties='monospace')
    axes[0].axis('off')

    # Seconds axis
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot(ax=axes[1], cmap='YlGn_r')
    axes[1].set_title('Confusion Matrix - Training Set')

    # Plot confusion matrix side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # First axis
    axes[0].text(0.01, 0.05, str(test_report), font_properties='monospace')
    axes[0].axis('off')

    # Seconds axis
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_train.plot(ax=axes[1], cmap='Purples')
    axes[1].set_title('Confusion Matrix - Testing Set')

    return train_report, test_report

    plt.show()

from sklearn.pipeline import Pipeline

# ML pipeline
def train_predict_model(X_train, y_train, X_test, preprocessor, model):

    # Combine pipeline and model
    model_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Fit pipeline on training data
    model_pipe.fit(X_train, y_train)

    # Save predictions
    train_preds = model_pipe.predict(X_train)
    test_preds = model_pipe.predict(X_test)

    return train_preds, test_preds

import joblib
import os

# Save model
def save_model(model, model_path):
    try:
        # Check if model directory exists
        dir_name = os.path.dirname(model_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # Save the model
        joblib.dump(model, model_path)

        # Confirm save if successful
        if os.path.exists(model_path):
            print(f'Model saved successfully to: {model_path}')
        else:
            print(f'Failed to save model: {model_path}')        

    except Exception as error:
        print(f'Error saving mode to {model_path}: {error}')


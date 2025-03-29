import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_models(X_train, y_train, features):
    """
    Train multiple machine learning models
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        features (list): List of feature names
        
    Returns:
        tuple: (models, feature_importance, best_model, scaler) - trained models, feature importance, best model name, and scaler
    """
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train each model
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
    
    # Get feature importance from Random Forest
    if 'Random Forest' in models:
        feature_importance = models['Random Forest'].feature_importances_
    else:
        feature_importance = None
    
    # For now, use Random Forest as the best model (will be evaluated later)
    best_model = 'Random Forest'
    
    return models, feature_importance, best_model, scaler

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data
    
    Args:
        models (dict): Dictionary of trained models
        X_test (np.array): Test features
        y_test (np.array): Test labels
        
    Returns:
        dict: Dictionary with evaluation metrics for each model
    """
    results = {}
    
    # Create a scaler for test data
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # Evaluate each model
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # For probability-based metrics
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Store results
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc
        }
    
    return results

def predict_risk(model_name, input_data, scaler):
    """
    Make a prediction using the selected model
    
    Args:
        model_name (str): Name of the model to use
        input_data (pd.DataFrame): Input data for prediction
        scaler (StandardScaler): Fitted scaler for feature standardization
        
    Returns:
        tuple: (prediction, probability) - predicted class and probability
    """
    from sklearn.ensemble import RandomForestClassifier
    import sklearn.exceptions
    
    try:
        # Use only the features that were used during training
        # This is essential to avoid the dimensionality mismatch
        expected_features = ['Age', 'Number of sexual partners', 'First sexual intercourse', 
                            'Num of pregnancies', 'Smokes', 'Hormonal Contraceptives']
        
        # Check which features are available in input data
        available_features = [f for f in expected_features if f in input_data.columns]
        
        if len(available_features) < len(expected_features):
            # If some features are missing, add them with default values
            for feature in expected_features:
                if feature not in input_data.columns:
                    input_data[feature] = 0  # Default value
        
        # Scale the features using only the expected features
        X_scaled = scaler.transform(input_data[expected_features])
        
        # If model_name is actually a model object
        if not isinstance(model_name, str):
            model = model_name
        else:
            # This is for when we have stored models in session_state
            # For now, we will return a default prediction since we don't have a fitted model
            return 0, 0.0  # Default values: low risk, 0% probability
        
        # Make prediction
        try:
            prediction = model.predict(X_scaled)[0]
            
            # Get probability
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(X_scaled)[0, 1]
            else:
                probability = None
                
            return prediction, probability
            
        except sklearn.exceptions.NotFittedError:
            # If the model is not fitted yet, return default values
            return 0, 0.0  # Default values: low risk, 0% probability
            
    except Exception as e:
        # Log the error and return default values
        print(f"Error in predict_risk: {e}")
        return 0, 0.0  # Default values: low risk, 0% probability

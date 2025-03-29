import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def handle_missing_values(data):
    """
    Handle missing values in the dataset
    
    Args:
        data (pd.DataFrame): The input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Replace '?' with NaN
    df = df.replace('?', np.nan)
    
    # Convert columns to appropriate data types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    
    # Handle numeric columns with SimpleImputer (mean imputation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df

def preprocess_data(data, features=None, target='Biopsy'):
    """
    Preprocess data for model training
    
    Args:
        data (pd.DataFrame): The input dataframe
        features (list): List of feature names to use
        target (str): Target variable name
        
    Returns:
        tuple: (X, y, feature_names, scaler) - preprocessed data, labels, feature names, and scaler
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Select features if provided, otherwise use all except target
    if features is None:
        features = [col for col in df.columns if col != target]
    
    # Check if target exists
    if target not in df.columns:
        return None, None, None, None
    
    # Remove rows where target is missing
    df = df.dropna(subset=[target])
    
    # Extract features and target
    X = df[features]
    y = df[target]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, features, scaler

def create_features(data, selected_features, target='Biopsy'):
    """
    Create feature matrix and target vector
    
    Args:
        data (pd.DataFrame): The input dataframe
        selected_features (list): List of selected feature names
        target (str): Target variable name
        
    Returns:
        tuple: (X, y, feature_names) - feature matrix, target vector, and feature names
    """
    # Check if target column exists
    if target not in data.columns:
        return None, None, None
    
    # Preprocess data
    X, y, features, _ = preprocess_data(data, selected_features, target)
    
    return X, y, features

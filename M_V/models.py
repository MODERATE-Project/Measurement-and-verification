from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, TimeSeriesSplit, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
import joblib
import time
from typing import List, Tuple, Dict, Any
import lightgbm as lgb
import optuna
import shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.layers import LeakyReLU
from sklearn.base import clone
import scipy.stats as stats
from sklearn.linear_model import LassoCV, Lasso, ElasticNet, RidgeCV, Ridge

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pickle
import os

def save_model_pickle(model, model_name, folder_path):
    '''
    Save model in pickle formart
    Param
    -----
    model: ML model
    mondel_name: name of the model
    folder_path: path to savel pickele model
    '''
    # Salvare il modello in formato pickle
    model_filename = f"{folder_path}/{model_name}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    print(f"\nModello salvato con successo in: {os.path.abspath(model_filename)}")

def data_structure_for_linaer_model(best_df_training,best_df_testing, best_df_reporting, target, best_features, type_regularization, model_name, folder_path):
    '''
    Save input to generate graph of best model
    '''
    data_structure = {
        "df_training": best_df_training, 
        "df_testing": best_df_testing, 
        "df_reporting": best_df_reporting, 
        "target_model": target,
        "features_model": best_features, 
        "regularization_type": type_regularization
    }

    # save inputs in picke format
    return save_model_pickle(
        model = data_structure,
        model_name = model_name,
        folder_path = folder_path
    )

def load_pickle_inputs(pickle_file_path):
    '''
    Load inputs from pickle file
    '''
    with open(pickle_file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    return loaded_data


def enhanced_linear_regression_noplot(
    df_training: pd.DataFrame, 
    df_testing: pd.DataFrame, 
    target: str, 
    features: list[str],
    polynomial_degree: int = None,
    handle_outliers: bool = True,
    cross_validation: bool = True,
    type_regularization: str = None,  # None, 'ridge', 'lasso', or 'elastic'
    n_folds: int = 5,
    time_series_cv: bool = True,
    permutation_importance: bool = True,
    confidence_interval: float = 0.95,
    verbose: int = 1,
    random_state: int = 42
) -> tuple:
    """
    Enhanced Linear Regression model with options for regularization, 
    cross-validation, outlier handling.
    
    Parameters:
    -----------
    df_training : pd.DataFrame
        Training dataframe
    df_testing : pd.DataFrame
        Testing dataframe
    target : str
        Target variable name
    features : list[str]
        List of feature names
    handle_outliers : bool, default=True
        Whether to handle outliers in training data
    cross_validation : bool, default=True
        Whether to use cross-validation for model evaluation
    type_regularization : str, default=None
        Type of regularization to apply ('ridge', 'lasso', 'elastic', or None)
    n_folds : int, default=5
        Number of folds for cross-validation
    time_series_cv : bool, default=True
        Whether to use time series split for cross-validation
    permutation_importance: bool, default=True
        Whether to calculate permutation feature importance
    verbose : int, default=1
        Verbosity level (0: silent, 1: important info)
    confidence_interval: float, default=0.95
        Confidence level for prediction intervals (0-1)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple containing:
    - Trained model
    - Test predictions Series
    - Training MAE
    - Training R²
    - Test MAE
    - Test R²
    - Additional diagnostics dictionary
    """
    start_time = time.time()
    # if verbose >= 1:
    #     print("\n" + "="*80)
    #     print(" Enhanced Linear Regression Model ")
    #     print("="*80)
    
    # Create copies to avoid modifying original data
    train_data = df_training.copy()
    test_data = df_testing.copy()
    
    # Extract features and target
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Handle outliers if requested
    weights = None
    if handle_outliers:
        q1 = y_train.quantile(0.25)
        q3 = y_train.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        
        outliers = (y_train > upper_bound) | (y_train < lower_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0 and outlier_count < len(y_train) * 0.1:  # Don't handle too many
            # if verbose >= 1:
            #     print(f"\nHandling {outlier_count} outliers ({outlier_count/len(y_train)*100:.2f}%)")
            
            # Create sample weights (lower for outliers)
            weights = np.ones(len(y_train))
            weights[outliers] = 0.3
    
    # Feature engineering and preprocessing
    if isinstance(polynomial_degree, int) and polynomial_degree > 1:
        if verbose >= 1:
            print(f"\nExpanding features using PolynomialFeatures (degree={polynomial_degree})...")
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        X_train_scaled = poly.fit_transform(X_train)
        X_test_scaled = poly.transform(X_test)
        expanded_feature_names = poly.get_feature_names_out(features)
    else:
        X_train_scaled = np.array(X_train)
        X_test_scaled = np.array(X_test)
        expanded_feature_names = features.copy()
    
    # Select model type based on regularization parameter
    if type_regularization == 'ridge':
        # if verbose >= 1:
        #     print("\nUsing Ridge Regression...")

        alphas = np.logspace(-3, 3, 20)

        if time_series_cv:
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        if cross_validation:
            # Find optimal alpha using RidgeCV
            model_cv = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_absolute_error')
            model_cv.fit(X_train_scaled, y_train, sample_weight=weights)
            alpha = model_cv.alpha_

            # if verbose >= 1:
            #     print(f"Selected alpha: {alpha:.6f}")
        else:
            alpha = 1.0

        # Fit the model with selected alpha
        model = Ridge(alpha=alpha, random_state=random_state)
        
        if cross_validation:
            maes = []
            r2s = []

            for train_idx, val_idx in cv.split(X_train_scaled):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                if weights is not None:
                    w_fold_train = weights[train_idx]
                else:
                    w_fold_train = None

                model_fold = Ridge(alpha=alpha, random_state=random_state)
                model_fold.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
                y_pred = model_fold.predict(X_fold_val)

                maes.append(mean_absolute_error(y_fold_val, y_pred))
                r2s.append(r2_score(y_fold_val, y_pred))

            cv_results = {
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "r2_mean": np.mean(r2s),
                "r2_std": np.std(r2s),
            }

            # if verbose >= 1:
            #     print(f"Cross-validation MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
            #     print(f"Cross-validation R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
            
    elif type_regularization == 'lasso':
        if verbose >= 1:
            print("\nUsing Lasso Regression...")
        
        if cross_validation:
            alphas = np.logspace(-3, 3, 20)
            if time_series_cv:
                cv = TimeSeriesSplit(n_splits=n_folds)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                
            model_cv = LassoCV(alphas=alphas, cv=cv, random_state=random_state)
            model_cv.fit(X_train_scaled, y_train)
            alpha = model_cv.alpha_
            
            if verbose >= 1:
                print(f"Selected alpha: {alpha:.6f}")
                
            model = Lasso(alpha=alpha, random_state=random_state)
        else:
            model = Lasso(alpha=1.0, random_state=random_state)
            
    elif type_regularization == 'elastic':
        if verbose >= 1:
            print("\nUsing ElasticNet Regression...")
        
        if cross_validation:
            # Create parameter grid
            param_grid = {
                'alpha': np.logspace(-3, 3, 10),
                'l1_ratio': np.linspace(0.1, 0.9, 9)
            }
            
            if time_series_cv:
                cv = TimeSeriesSplit(n_splits=n_folds)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                
            grid = GridSearchCV(
                ElasticNet(random_state=random_state), 
                param_grid, 
                cv=cv, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid.fit(X_train_scaled, y_train, sample_weight=weights)
            
            alpha = grid.best_params_['alpha']
            l1_ratio = grid.best_params_['l1_ratio']
            
            if verbose >= 1:
                print(f"Selected alpha: {alpha:.6f}, l1_ratio: {l1_ratio:.2f}")
                
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        else:
            model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state)
    else:
        if verbose >= 1:
            print("\nUsing standard Linear Regression...")
        model = LinearRegression()
    
    # Train the model
    model.fit(X_train_scaled, y_train, sample_weight=weights)
    
    # Cross-validation evaluation
    cv_results = {}
    if cross_validation:
        if time_series_cv:
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            
        mae_scores = []
        r2_scores = []

        for train_idx, val_idx in cv.split(X_train_scaled):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if weights is not None:
                w_fold_train = weights[train_idx]
            else:
                w_fold_train = None

            model_fold = clone(model)
            model_fold.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
            
            y_pred = model_fold.predict(X_fold_val)
            
            mae_scores.append(mean_absolute_error(y_fold_val, y_pred))
            r2_scores.append(r2_score(y_fold_val, y_pred))

        cv_results = {
            'cv_mae_mean': np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores),
            'cv_r2_mean': np.mean(r2_scores),
            'cv_r2_std': np.std(r2_scores)
        }
        
        # if verbose >= 1:
        #     print("\nCross-Validation Results:")
        #     print(f"  MAE: {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f}")
        #     print(f"  R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Create DataFrame with predictions and actual values
    df_results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test,
        'index': X_test.index
    })
    
    # Calculate confidence intervals for predictions
    # For linear regression, we'll use the standard error of the residuals
    # to estimate prediction intervals
    
    # # Calculate residuals on training data
    # residuals = y_train - y_pred_train
    
    # # Calculate standard error of residuals
    # residual_std = np.std(residuals)
    
    # # Calculate critical value for the desired confidence level
    
    # alpha = 1 - confidence_interval
    # critical_value = stats.t.ppf(1 - alpha/2, df=len(y_train) - len(features) - 1)
    
    # # Calculate prediction interval
    # margin_of_error = critical_value * residual_std
    
    # # Calculate lower and upper bounds for test predictions
    # lower_bound = y_pred_test - margin_of_error
    # upper_bound = y_pred_test + margin_of_error
    
    # # Add confidence intervals to DataFrame
    # df_results['lower_bound'] = lower_bound
    # df_results['upper_bound'] = upper_bound
    
    # Residui
    try: 
        residui = y_train - y_pred_train
        mse = np.mean(residui**2)

        # Calcolo della varianza della predizione
        X_design = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])  # aggiungiamo intercetta
        cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)

        # Scelta livello di confidenza (es. 95%)
        alpha = 0.05
        t_value = stats.t.ppf(1 - alpha/2, df=len(y_train) - X_design.shape[1])

        # Aggiungiamo colonna di 1 per l'intercetta a X_test
        X_test_design = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

        # Calcolo della varianza della predizione punto per punto
        pred_var = np.sum(X_test_design @ cov_matrix * X_test_design, axis=1)
        margin = t_value * np.sqrt(pred_var)

        # Intervallo di confidenza
        lower_bound = y_pred_test - margin
        upper_bound = y_pred_test + margin

        # Add confidence intervals to DataFrame
        df_results['lower_bound'] = lower_bound
        df_results['upper_bound'] = upper_bound

    except:
        df_results['lower_bound'] = np.NaN
        df_results['upper_bound'] = np.NaN

    # Ensure predictions are non-negative if target is strictly positive
    if y_train.min() >= 0:
        df_results['predicted'] = np.maximum(0, df_results['predicted'])
        y_pred_train = np.maximum(0, y_pred_train)
    
    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(df_results['actual'], df_results['predicted'])
    rmse_test = np.sqrt(mean_squared_error(df_results['actual'], df_results['predicted']))
    r2_test = r2_score(df_results['actual'], df_results['predicted'])
    
    # Calculate overfitting metrics
    overfitting_ratio_mae = mae_test / mae_train if mae_train > 0 else float('inf')
    overfitting_ratio_rmse = rmse_test / rmse_train if rmse_train > 0 else float('inf')
    
    # Calculate prediction interval coverage
    within_interval = ((df_results['actual'] >= df_results['lower_bound']) & 
                       (df_results['actual'] <= df_results['upper_bound']))
    interval_coverage = within_interval.mean()
    
    if verbose >= 1:
        print("\nModel Performance:")
        print(f"  Training Set:")
        print(f"    - MAE: {mae_train:.4f}")
        print(f"    - RMSE: {rmse_train:.4f}")
        print(f"    - R²: {r2_train:.4f}")
        print(f"  Test Set:")
        print(f"    - MAE: {mae_test:.4f}")
        print(f"    - RMSE: {rmse_test:.4f}")
        print(f"    - R²: {r2_test:.4f}")
        print(f"  Overfitting Assessment:")
        print(f"    - MAE Ratio (Test/Train): {overfitting_ratio_mae:.2f} (>1.3 indicates overfitting)")
        print(f"    - RMSE Ratio (Test/Train): {overfitting_ratio_rmse:.2f} (>1.3 indicates overfitting)")
        print(f"  Prediction Interval Coverage: {interval_coverage:.4f} (expected: {confidence_interval:.4f})")
    
    # Extract feature coefficients
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': expanded_feature_names,
            'Coefficient': model.coef_
        })
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        # if verbose >= 1:
        #     print("\nFeature Coefficients:")
        #     for i, row in coef_df.head(10).iterrows():
        #         print(f"  - {row['Feature']}: {row['Coefficient']:.6f}")
    else:
        coef_df = None
    
    # Calculate permutation importance
    perm_importance_df = None
    if permutation_importance:
        # if verbose >= 1:
        #     print("\nCalculating permutation feature importance...")
        
        # We'll use scikit-learn's permutation_importance function
        from sklearn.inspection import permutation_importance
        
        # Calculate on both train and test sets
        perm_importance_train = permutation_importance(
            model, X_train_scaled, y_train, 
            n_repeats=10, 
            random_state=random_state,
            scoring='neg_mean_absolute_error'
        )
        
        perm_importance_test = permutation_importance(
            model, X_test_scaled, y_test, 
            n_repeats=10, 
            random_state=random_state,
            scoring='neg_mean_absolute_error'
        )
        
        # Create DataFrames
        perm_importance_train_df = pd.DataFrame({
            'Feature': expanded_feature_names,
            'Train_Importance_Mean': perm_importance_train.importances_mean,
            'Train_Importance_Std': perm_importance_train.importances_std
        })
        
        perm_importance_test_df = pd.DataFrame({
            'Feature': expanded_feature_names,
            'Test_Importance_Mean': perm_importance_test.importances_mean,
            'Test_Importance_Std': perm_importance_test.importances_std
        })
        
        # Merge the DataFrames
        perm_importance_df = pd.merge(
            perm_importance_train_df, 
            perm_importance_test_df, 
            on='Feature'
        )
        
        # Sort by test importance (absolute value)
        perm_importance_df['Abs_Test_Importance'] = perm_importance_df['Test_Importance_Mean'].abs()
        perm_importance_df = perm_importance_df.sort_values('Abs_Test_Importance', ascending=False)
        
        # if verbose >= 1:
        #     print("\nPermutation Feature Importance (Top 10):")
        #     for i, row in perm_importance_df.head(10).iterrows():
        #         print(f"  - {row['Feature']}: {row['Test_Importance_Mean']:.6f} ± {row['Test_Importance_Std']:.6f}")

    # Record execution time
    execution_time = time.time() - start_time
    # if verbose >= 1:
    #     print(f"\nExecution completed in {execution_time:.2f} seconds")
    
    # Prepare output series
    pred_test_series = pd.Series(df_results['predicted'].values, index=df_results['index'])
    
    # Create diagnostics dictionary
    diagnostics = {
        'model_type': type(model).__name__,
        'coef_df': coef_df,
        'permutation_importance_df': perm_importance_df,
        'cv_results': cv_results,
        'scaled_features': True,
        'metrics': {
            'mae_train': mae_train,
            'rmse_train': rmse_train,
            'r2_train': r2_train,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test,
            'overfitting_ratio_mae': overfitting_ratio_mae,
            'overfitting_ratio_rmse': overfitting_ratio_rmse,
            'prediction_interval_coverage': interval_coverage
        },
        'execution_time': execution_time,
        # 'feature_scaler': scaler
    }
    
    return model, pred_test_series, mae_train, r2_train, mae_test, r2_test, diagnostics, coef_df




# def generate_regression_plots(
#     model,
#     X_train, 
#     y_train, 
#     X_test, 
#     y_test,
#     features,
#     target,
#     predictions_test=None,
#     scaler=None,
#     confidence_interval=0.95,
#     permutation_imp=True,
#     random_state=42
# ):
#     """
#     Generate comprehensive plots for a regression model.
    
#     Parameters:
#     -----------
#     model : trained regression model
#         The trained regression model
#     X_train : pd.DataFrame
#         Training features
#     y_train : pd.Series
#         Training target
#     X_test : pd.DataFrame
#         Testing features
#     y_test : pd.Series
#         Testing target
#     features : list
#         List of feature names
#     target : str
#         Name of the target variable
#     predictions_test : pd.Series, optional
#         Precomputed test predictions. If None, they will be computed
#     scaler : sklearn scaler, optional
#         Scaler used for feature scaling (if any)
#     confidence_interval : float, default=0.95
#         Confidence level for prediction intervals (0-1)
#     permutation_imp : bool, default=True
#         Whether to calculate and plot permutation importance
#     random_state : int, default=42
#         Random seed for reproducibility
        
#     Returns:
#     --------
#     None (displays plots)
#     """
#     # Scale features if scaler is provided
#     if scaler is not None:
#         X_train_scaled = scaler.transform(X_train)
#         X_test_scaled = scaler.transform(X_test)
#     else:
#         X_train_scaled = X_train.values
#         X_test_scaled = X_test.values
    
#     # Make predictions if not provided
#     if predictions_test is None:
#         y_pred_train = model.predict(X_train_scaled)
#         y_pred_test = model.predict(X_test_scaled)
#     else:
#         y_pred_test = predictions_test
#         y_pred_train = model.predict(X_train_scaled)
    
#     # Create DataFrame with predictions and actual values
#     df_results = pd.DataFrame({
#         'actual': y_test,
#         'predicted': y_pred_test,
#         'index': X_test.index
#     })
    
#     # Calculate residuals
#     residuals_train = y_train - y_pred_train
#     residuals_test = df_results['actual'] - df_results['predicted']
    
#     # Calculate confidence intervals
#     residual_std = np.std(residuals_train)
#     alpha = 1 - confidence_interval
#     critical_value = stats.t.ppf(1 - alpha/2, df=len(y_train) - len(features) - 1)
#     margin_of_error = critical_value * residual_std
    
#     lower_bound = df_results['predicted'] - margin_of_error
#     upper_bound = df_results['predicted'] + margin_of_error
    
#     df_results['lower_bound'] = lower_bound
#     df_results['upper_bound'] = upper_bound
    
#     # Calculate interval coverage
#     within_interval = ((df_results['actual'] >= df_results['lower_bound']) & 
#                        (df_results['actual'] <= df_results['upper_bound']))
#     interval_coverage = within_interval.mean()
#     print(f"Prediction interval coverage: {interval_coverage:.4f} (expected: {confidence_interval:.4f})")
    
#     # 1. Actual vs Predicted Plot with Confidence Intervals (Plotly)
#     print("\nGenerating Actual vs Predicted plot with confidence intervals...")
#     fig = go.Figure()

#     # Add confidence interval as a filled area
#     fig.add_trace(go.Scatter(
#         x=df_results['index'].tolist() + df_results['index'].tolist()[::-1],
#         y=df_results['upper_bound'].tolist() + df_results['lower_bound'].tolist()[::-1],
#         fill='toself',
#         fillcolor='rgba(0,176,246,0.2)',
#         line=dict(color='rgba(255,255,255,0)'),
#         name=f'{confidence_interval*100}% Confidence Interval'
#     ))
    
#     # Add actual values
#     fig.add_trace(go.Scatter(
#         x=df_results['index'], 
#         y=df_results['actual'], 
#         mode='lines+markers',
#         name='Actual',
#         marker=dict(size=6, color='blue'),
#         line=dict(width=2)
#     ))
    
#     # Add predicted values
#     fig.add_trace(go.Scatter(
#         x=df_results['index'], 
#         y=df_results['predicted'], 
#         mode='lines+markers',
#         name='Predicted',
#         marker=dict(size=6, color='red'),
#         line=dict(width=2, dash='dash')
#     ))
    
#     # Update layout
#     fig.update_layout(
#         title='Actual vs Predicted Values',
#         xaxis_title='Index',
#         yaxis_title=target,
#         legend=dict(x=0.01, y=0.99),
#         height=600,
#         width=1000
#     )
    
#     fig.show()
    
#     # 2. Multiple Diagnostic Plots
#     print("\nGenerating diagnostic plots...")
    
#     # Adjust number of rows based on whether we have permutation importance
#     if permutation_imp and hasattr(model, 'coef_'):
#         fig, axes = plt.subplots(3, 2, figsize=(16, 18))
#     else:
#         fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
#     # Plot 1: Actual vs Predicted Scatter
#     axes[0, 0].scatter(df_results['actual'], df_results['predicted'], alpha=0.6)
    
#     # Add 45-degree line
#     min_val = min(df_results['actual'].min(), df_results['predicted'].min())
#     max_val = max(df_results['actual'].max(), df_results['predicted'].max())
#     axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
#     axes[0, 0].set_xlabel('Actual Values')
#     axes[0, 0].set_ylabel('Predicted Values')
#     axes[0, 0].set_title('Actual vs Predicted Values (Test Set)')
    
#     # Plot 2: Residuals distribution
#     sns.histplot(residuals_test, kde=True, ax=axes[0, 1], color='blue', alpha=0.6)
#     axes[0, 1].axvline(x=0, color='red', linestyle='--')
#     axes[0, 1].set_xlabel('Residuals (Actual - Predicted)')
#     axes[0, 1].set_ylabel('Frequency')
#     axes[0, 1].set_title('Residuals Distribution')
    
#     # Plot 3: Feature importance (coefficients) if applicable
#     if hasattr(model, 'coef_'):
#         # Create coefficient DataFrame
#         coef_df = pd.DataFrame({
#             'Feature': features,
#             'Coefficient': model.coef_
#         })
#         coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
#         coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
#         top_features = coef_df.head(10)
#         sns.barplot(x='Coefficient', y='Feature', data=top_features, ax=axes[1, 0])
#         axes[1, 0].set_title('Top 10 Features by Coefficient Magnitude')
#     else:
#         axes[1, 0].text(0.5, 0.5, 'No coefficient data available', 
#                         horizontalalignment='center', verticalalignment='center')
    
#     # Plot 4: Residuals vs Predicted
#     axes[1, 1].scatter(df_results['predicted'], residuals_test, alpha=0.6)
#     axes[1, 1].axhline(y=0, color='red', linestyle='--')
#     axes[1, 1].set_xlabel('Predicted Values')
#     axes[1, 1].set_ylabel('Residuals')
#     axes[1, 1].set_title('Residuals vs Predicted Values')
    
#     # Add Permutation Importance Plot if requested
#     if permutation_imp and hasattr(model, 'coef_'):
#         print("\nCalculating permutation feature importance...")
        
#         # Calculate permutation importance on both train and test sets
#         perm_importance_train = permutation_importance(
#             model, X_train_scaled, y_train, 
#             n_repeats=10, 
#             random_state=random_state,
#             scoring='neg_mean_absolute_error'
#         )
        
#         perm_importance_test = permutation_importance(
#             model, X_test_scaled, y_test, 
#             n_repeats=10, 
#             random_state=random_state,
#             scoring='neg_mean_absolute_error'
#         )
        
#         # Create DataFrames
#         perm_importance_train_df = pd.DataFrame({
#             'Feature': features,
#             'Train_Importance_Mean': perm_importance_train.importances_mean,
#             'Train_Importance_Std': perm_importance_train.importances_std
#         })
        
#         perm_importance_test_df = pd.DataFrame({
#             'Feature': features,
#             'Test_Importance_Mean': perm_importance_test.importances_mean,
#             'Test_Importance_Std': perm_importance_test.importances_std
#         })
        
#         # Merge the DataFrames
#         perm_importance_df = pd.merge(
#             perm_importance_train_df, 
#             perm_importance_test_df, 
#             on='Feature'
#         )
        
#         # Sort by test importance (absolute value)
#         perm_importance_df['Abs_Test_Importance'] = perm_importance_df['Test_Importance_Mean'].abs()
#         perm_importance_df = perm_importance_df.sort_values('Abs_Test_Importance', ascending=False)
        
#         # Plot 5: Permutation Importance (Train)
#         top_perm_features = perm_importance_df.head(10)
#         sns.barplot(
#             x='Train_Importance_Mean', 
#             y='Feature', 
#             data=top_perm_features, 
#             ax=axes[2, 0],
#             color='skyblue'
#         )
#         axes[2, 0].set_title('Top 10 Features by Permutation Importance (Training Set)')
#         axes[2, 0].set_xlabel('Decrease in Performance')
        
#         # Plot 6: Permutation Importance (Test)
#         sns.barplot(
#             x='Test_Importance_Mean', 
#             y='Feature', 
#             data=top_perm_features, 
#             ax=axes[2, 1],
#             color='lightgreen'
#         )
#         axes[2, 1].set_title('Top 10 Features by Permutation Importance (Test Set)')
#         axes[2, 1].set_xlabel('Decrease in Performance')
    
#     plt.tight_layout()
#     plt.show()
    
#     # 3. QQ Plot for residuals normality check
#     print("\nGenerating Q-Q plot of residuals...")
#     plt.figure(figsize=(8, 8))
#     stats.probplot(residuals_test, dist="norm", plot=plt)
#     plt.title('Q-Q Plot of Residuals')
#     plt.tight_layout()
#     plt.show()
    
#     # 4. Confidence interval coverage plot
#     print("\nGenerating confidence interval coverage plot...")
#     plt.figure(figsize=(12, 6))
    
#     # Sort results by actual values for better visualization
#     sorted_idx = np.argsort(df_results['actual'])
#     sorted_actual = df_results['actual'].iloc[sorted_idx]
#     sorted_predicted = df_results['predicted'].iloc[sorted_idx]
#     sorted_lower = df_results['lower_bound'].iloc[sorted_idx]
#     sorted_upper = df_results['upper_bound'].iloc[sorted_idx]
#     sorted_within = within_interval.iloc[sorted_idx]
    
#     # Plot actual values
#     plt.plot(range(len(sorted_actual)), sorted_actual, 'b-', label='Actual')
    
#     # Plot predicted values
#     plt.plot(range(len(sorted_predicted)), sorted_predicted, 'r--', label='Predicted')
    
#     # Plot confidence intervals
#     plt.fill_between(
#         range(len(sorted_actual)),
#         sorted_lower,
#         sorted_upper,
#         color='gray', alpha=0.3,
#         label=f'{confidence_interval*100}% Confidence Interval'
#     )
    
#     # Highlight points outside the confidence interval
#     outside_interval = ~sorted_within
#     outside_interval_indices = np.where(outside_interval.values)[0]
#     if len(outside_interval_indices) > 0:
#         plt.scatter(
#             outside_interval_indices,
#             sorted_actual.iloc[outside_interval_indices],
#             color='red', s=50, zorder=5,
#             label='Outside Confidence Interval'
#         )
    
#     plt.title(f'Predictions with {confidence_interval*100}% Confidence Intervals')
#     plt.xlabel('Sorted Sample Index')
#     plt.ylabel(target)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# # Function to easily create plots from an enhanced_linear_regression model result
# def plot_enhanced_regression_results(model_result, df_training, df_testing, target, features):
#     """
#     Generate plots from the results of enhanced_linear_regression function.
    
#     Parameters:
#     -----------
#     model_result : tuple
#         The tuple returned by enhanced_linear_regression function
#     df_training : pd.DataFrame
#         Training dataframe used to train the model
#     df_testing : pd.DataFrame
#         Testing dataframe used to test the model
#     target : str
#         Target variable name
#     features : list[str]
#         List of feature names used for training
        
#     Returns:
#     --------
#     None (displays plots)
#     """
#     model, predictions, mae_train, r2_train, mae_test, r2_test, diagnostics, coef_df = model_result
    
#     # Extract the scaler from diagnostics
#     scaler = diagnostics.get('feature_scaler', None)
    
#     # Extract X and y from dataframes
#     X_train = df_training[features]
#     y_train = df_training[target]
#     X_test = df_testing[features]
#     y_test = df_testing[target]
    
#     # Generate all plots
#     generate_regression_plots(
#         model=model,
#         X_train=X_train,
#         y_train=y_train,
#         X_test=X_test,
#         y_test=y_test,
#         features=features,
#         target=target,
#         predictions_test=predictions,
#         scaler=scaler
#     )
    
#     # Print key metrics
#     print("\nModel Performance Summary:")
#     print(f"Training MAE: {mae_train:.4f}")
#     print(f"Training R²: {r2_train:.4f}")
#     print(f"Test MAE: {mae_test:.4f}")
#     print(f"Test R²: {r2_test:.4f}")
    
#     # Print overfitting metrics
#     if 'metrics' in diagnostics:
#         metrics = diagnostics['metrics']
#         print(f"\nOverfitting Assessment:")
#         print(f"MAE Ratio (Test/Train): {metrics.get('overfitting_ratio_mae', 'N/A')}")
#         print(f"RMSE Ratio (Test/Train): {metrics.get('overfitting_ratio_rmse', 'N/A')}")


#     return model, y_pred_test, mae_train, r2_train, mae_test, r2_test
def enhanced_linear_regression(
    df_training: pd.DataFrame, 
    df_testing: pd.DataFrame, 
    df_reporting: pd.DataFrame,
    target: str, 
    features: list[str],
    polynomial_degree:int = None,
    handle_outliers: bool = True,
    cross_validation: bool = True,
    type_regularization: str = None,  # None, 'ridge', 'lasso', or 'elastic'
    n_folds: int = 5,
    time_series_cv: bool = True,
    plot_results: bool = True,
    plot_diagnostics: bool = True,
    permutation_importance: bool = True,  # Added parameter for permutation importance
    confidence_interval: float = 0.95,  # Added confidence interval parameter
    verbose: int = 1,
    random_state: int = 42
) -> tuple:
    """
    Enhanced Linear Regression model with options for regularization, 
    cross-validation, outlier handling, and diagnostic plots.
    
    Parameters:
    -----------
    df_training : pd.DataFrame
        Training dataframe
    df_testing : pd.DataFrame
        Testing dataframe
    target : str
        Target variable name
    features : list[str]
        List of feature names
    handle_outliers : bool, default=True
        Whether to handle outliers in training data
    cross_validation : bool, default=True
        Whether to use cross-validation for model evaluation
    type_regularization : str, default=None
        Type of regularization to apply ('ridge', 'lasso', 'elastic', or None)
    n_folds : int, default=5
        Number of folds for cross-validation
    time_series_cv : bool, default=True
        Whether to use time series split for cross-validation
    plot_results : bool, default=True
        Whether to plot test predictions vs actual values
    plot_diagnostics : bool, default=True
        Whether to plot diagnostic plots (residuals, etc.)
    permutation_importance: bool, default=True
        Whether to calculate and visualize permutation feature importance
    verbose : int, default=1
        Verbosity level (0: silent, 1: important info)
    confidence_interval: float, default=0.95
        Confidence level for prediction intervals (0-1)
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    tuple containing:
    - Trained model
    - Test predictions Series
    - Training MAE
    - Training R²
    - Test MAE
    - Test R²
    - Additional diagnostics dictionary
    """
    start_time = time.time()
    if verbose >= 1:
        print("\n" + "="*80)
        print(" Enhanced Linear Regression Model ")
        print("="*80)
    
    # Create copies to avoid modifying original data
    train_data = df_training.copy()
    test_data = df_testing.copy()
    reporting_data = df_reporting.copy()
    
    # Extract features and target
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    X_reporting = reporting_data[features]
    y_reporting = reporting_data[target]
    
    features_ = X_train.columns
    # Handle outliers if requested
    weights = None
    if handle_outliers:
        q1 = y_train.quantile(0.25)
        q3 = y_train.quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        
        outliers = (y_train > upper_bound) | (y_train < lower_bound)
        outlier_count = outliers.sum()
        
        if outlier_count > 0 and outlier_count < len(y_train) * 0.1:  # Don't handle too many
            if verbose >= 1:
                print(f"\nHandling {outlier_count} outliers ({outlier_count/len(y_train)*100:.2f}%)")
            
            # Create sample weights (lower for outliers)
            weights = np.ones(len(y_train))
            weights[outliers] = 0.3
    
    # Feature engineering and preprocessing
    # Feature engineering and preprocessing
    if isinstance(polynomial_degree, int) and polynomial_degree > 1:
        if verbose >= 1:
            print(f"\nExpanding features using PolynomialFeatures (degree={polynomial_degree})...")
        poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        X_train_scaled = poly.fit_transform(X_train)
        X_test_scaled = poly.transform(X_test)
        X_reporting_scaled = poly.transform(X_reporting)
        expanded_feature_names = poly.get_feature_names_out(features)
    else:
        X_train_scaled = np.array(X_train)
        X_test_scaled = np.array(X_test)
        X_reporting_scaled = np.array(X_reporting)
        expanded_feature_names = features.copy()
    # # scaler = StandardScaler()
    # X_train_scaled = np.array(X_train)
    # # X_train_scaled = scaler.fit_transform(X_train)
    # # X_test_scaled = X_test
    # X_test_scaled = np.array(X_test)
    # # X_reporting_scaled = scaler.transform(X_reporting)
    # X_reporting_scaled = np.array(X_reporting)
    
    # Select model type based on regularization parameter
    if type_regularization == 'ridge':
        if verbose >= 1:
            print("\nUsing Ridge Regression...")

        alphas = np.logspace(-3, 3, 20)

        if time_series_cv:
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        if cross_validation:
            # Find optimal alpha using RidgeCV
            model_cv = RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_absolute_error')
            model_cv.fit(X_train_scaled, y_train, sample_weight=weights)
            alpha = model_cv.alpha_

            if verbose >= 1:
                print(f"Selected alpha: {alpha:.6f}")
        else:
            alpha = 1.0

        # Fit the model with selected alpha
        model = Ridge(alpha=alpha, random_state=random_state)
        
        if cross_validation:
            maes = []
            r2s = []

            for train_idx, val_idx in cv.split(X_train_scaled):
                X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                if weights is not None:
                    w_fold_train = weights[train_idx]
                else:
                    w_fold_train = None

                model_fold = Ridge(alpha=alpha, random_state=random_state)
                model_fold.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
                y_pred = model_fold.predict(X_fold_val)

                maes.append(mean_absolute_error(y_fold_val, y_pred))
                r2s.append(r2_score(y_fold_val, y_pred))

            cv_results = {
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "r2_mean": np.mean(r2s),
                "r2_std": np.std(r2s),
            }

            if verbose >= 1:
                print(f"Cross-validation MAE: {cv_results['mae_mean']:.4f} ± {cv_results['mae_std']:.4f}")
                print(f"Cross-validation R²: {cv_results['r2_mean']:.4f} ± {cv_results['r2_std']:.4f}")
            
    elif type_regularization == 'lasso':
        if verbose >= 1:
            print("\nUsing Lasso Regression...")
        
        if cross_validation:
            alphas = np.logspace(-3, 3, 20)
            if time_series_cv:
                cv = TimeSeriesSplit(n_splits=n_folds)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                
            model_cv = LassoCV(alphas=alphas, cv=cv, random_state=random_state)
            model_cv.fit(X_train_scaled, y_train)
            alpha = model_cv.alpha_
            
            if verbose >= 1:
                print(f"Selected alpha: {alpha:.6f}")
                
            model = Lasso(alpha=alpha, random_state=random_state)
        else:
            model = Lasso(alpha=1.0, random_state=random_state)
            
    elif type_regularization == 'elastic':
        if verbose >= 1:
            print("\nUsing ElasticNet Regression...")
        
        if cross_validation:
            # Create parameter grid
            param_grid = {
                'alpha': np.logspace(-3, 3, 10),
                'l1_ratio': np.linspace(0.1, 0.9, 9)
            }
            
            if time_series_cv:
                cv = TimeSeriesSplit(n_splits=n_folds)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
                
            grid = GridSearchCV(
                ElasticNet(random_state=random_state), 
                param_grid, 
                cv=cv, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid.fit(X_train_scaled, y_train, sample_weight=weights)
            
            alpha = grid.best_params_['alpha']
            l1_ratio = grid.best_params_['l1_ratio']
            
            if verbose >= 1:
                print(f"Selected alpha: {alpha:.6f}, l1_ratio: {l1_ratio:.2f}")
                
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        else:
            model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state)
    else:
        if verbose >= 1:
            print("\nUsing standard Linear Regression...")
        model = LinearRegression()
    
    # Train the model
    model.fit(X_train_scaled, y_train, sample_weight=weights)
    
    # Cross-validation evaluation
    cv_results = {}
    if cross_validation:
        if time_series_cv:
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            
        mae_scores = []
        r2_scores = []

        for train_idx, val_idx in cv.split(X_train_scaled):
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if weights is not None:
                w_fold_train = weights[train_idx]
            else:
                w_fold_train = None

            model_fold = clone(model)
            model_fold.fit(X_fold_train, y_fold_train, sample_weight=w_fold_train)
            
            y_pred = model_fold.predict(X_fold_val)
            
            mae_scores.append(mean_absolute_error(y_fold_val, y_pred))
            r2_scores.append(r2_score(y_fold_val, y_pred))

        cv_results = {
            'cv_mae_mean': np.mean(mae_scores),
            'cv_mae_std': np.std(mae_scores),
            'cv_r2_mean': np.mean(r2_scores),
            'cv_r2_std': np.std(r2_scores)
        }
        
        if verbose >= 1:
            print("\nCross-Validation Results:")
            print(f"  MAE: {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f}")
            print(f"  R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
    # else:
    #     cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    #     cv_mae = -cross_val_score(
    #         model, X_train_scaled, y_train, 
    #         cv=cv, scoring='neg_mean_absolute_error', 
    #         sample_weight=weights
    #     )
        
    #     cv_r2 = cross_val_score(
    #         model, X_train_scaled, y_train, 
    #         cv=cv, scoring='r2', 
    #         sample_weight=weights
    #     )
        
    #     cv_results = {
    #         'cv_mae_mean': cv_mae.mean(),
    #         'cv_mae_std': cv_mae.std(),
    #         'cv_r2_mean': cv_r2.mean(),
    #         'cv_r2_std': cv_r2.std()
    #     }
        
    #     if verbose >= 1:
    #         print("\nCross-Validation Results:")
    #         print(f"  MAE: {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f}")
    #         print(f"  R²: {cv_results['cv_r2_mean']:.4f} ± {cv_results['cv_r2_std']:.4f}")
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_pred_reporting = model.predict(X_reporting_scaled)
    
    # Create DataFrame with predictions and actual values
    df_results = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred_test,
        'index': X_test.index
    })
    df_results_train = pd.DataFrame({
        'actual_train': y_train,
        'predicted_train': y_pred_train,
    })
    df_results_reporting = pd.DataFrame({
        'actual_reporting': y_reporting,
        'predicted_reporting': y_pred_reporting,
    })
    
    # Calculate confidence intervals for predictions
    # For linear regression, we'll use the standard error of the residuals
    # to estimate prediction intervals
    
    # Calculate residuals on training data
    # residuals = y_train - y_pred_train
    
    # # Calculate standard error of residuals
    # residual_std = np.std(residuals)
    
    # # Calculate critical value for the desired confidence level
    # from scipy import stats
    # alpha = 1 - confidence_interval
    # critical_value = stats.t.ppf(1 - alpha/2, df=len(y_train) - len(features) - 1)
    
    # # Calculate prediction interval
    # margin_of_error = critical_value * residual_std
    
    # # Calculate lower and upper bounds for test predictions
    # lower_bound = y_pred_test - margin_of_error
    # upper_bound = y_pred_test + margin_of_error
    
    # # Add confidence intervals to DataFrame
    # df_results['lower_bound'] = lower_bound
    # df_results['upper_bound'] = upper_bound

    # Residui
    residui = y_train - y_pred_train
    mse = np.mean(residui**2)

    # Calcolo della varianza della predizione
    X_design = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])  # aggiungiamo intercetta
    cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)

    # Scelta livello di confidenza (es. 95%)
    alpha = 0.05
    t_value = stats.t.ppf(1 - alpha/2, df=len(y_train) - X_design.shape[1])

    # Aggiungiamo colonna di 1 per l'intercetta a X_test
    X_test_design = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

    # Calcolo della varianza della predizione punto per punto
    pred_var = np.sum(X_test_design @ cov_matrix * X_test_design, axis=1)
    margin = t_value * np.sqrt(pred_var)

    # Intervallo di confidenza
    lower_bound = y_pred_test - margin
    upper_bound = y_pred_test + margin

    # Add confidence intervals to DataFrame
    df_results['lower_bound'] = lower_bound
    df_results['upper_bound'] = upper_bound

    
    # Ensure predictions are non-negative if target is strictly positive
    if y_train.min() >= 0:
        df_results['predicted'] = np.maximum(0, df_results['predicted'])
        y_pred_train = np.maximum(0, y_pred_train)
    
    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    
    mae_test = mean_absolute_error(df_results['actual'], df_results['predicted'])
    rmse_test = np.sqrt(mean_squared_error(df_results['actual'], df_results['predicted']))
    r2_test = r2_score(df_results['actual'], df_results['predicted'])
    
    # Calculate overfitting metrics
    overfitting_ratio_mae = mae_test / mae_train if mae_train > 0 else float('inf')
    overfitting_ratio_rmse = rmse_test / rmse_train if rmse_train > 0 else float('inf')
    
    # Calculate prediction interval coverage
    within_interval = ((df_results['actual'] >= df_results['lower_bound']) & 
                       (df_results['actual'] <= df_results['upper_bound']))
    interval_coverage = within_interval.mean()
    if verbose >= 1:
        print("\nModel Performance:")
        print(f"  Training Set:")
        print(f"    - MAE: {mae_train:.4f}")
        print(f"    - RMSE: {rmse_train:.4f}")
        print(f"    - R²: {r2_train:.4f}")
        print(f"  Test Set:")
        print(f"    - MAE: {mae_test:.4f}")
        print(f"    - RMSE: {rmse_test:.4f}")
        print(f"    - R²: {r2_test:.4f}")
        print(f"  Overfitting Assessment:")
        print(f"    - MAE Ratio (Test/Train): {overfitting_ratio_mae:.2f} (>1.3 indicates overfitting)")
        print(f"    - RMSE Ratio (Test/Train): {overfitting_ratio_rmse:.2f} (>1.3 indicates overfitting)")
    
    # Extract feature coefficients
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({
            'Feature': expanded_feature_names,
            'Coefficient': model.coef_
        })
        coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        if verbose >= 1:
            print("\nFeature Coefficients:")
            for i, row in coef_df.head(10).iterrows():
                print(f"  - {row['Feature']}: {row['Coefficient']:.6f}")
    else:
        coef_df = None
    
    # Calculate permutation importance
    perm_importance_df = None
    if permutation_importance:
        if verbose >= 1:
            print("\nCalculating permutation feature importance...")
        
        # We'll use scikit-learn's permutation_importance function
        from sklearn.inspection import permutation_importance
        
        # Calculate on both train and test sets
        perm_importance_train = permutation_importance(
            model, X_train_scaled, y_train, 
            n_repeats=10, 
            random_state=random_state,
            scoring='neg_mean_absolute_error'
        )
        
        perm_importance_test = permutation_importance(
            model, X_test_scaled, y_test, 
            n_repeats=10, 
            random_state=random_state,
            scoring='neg_mean_absolute_error'
        )
        
        # Create DataFrames
        perm_importance_train_df = pd.DataFrame({
            'Feature': expanded_feature_names,
            'Train_Importance_Mean': perm_importance_train.importances_mean,
            'Train_Importance_Std': perm_importance_train.importances_std
        })
        
        perm_importance_test_df = pd.DataFrame({
            'Feature': expanded_feature_names,
            'Test_Importance_Mean': perm_importance_test.importances_mean,
            'Test_Importance_Std': perm_importance_test.importances_std
        })
        
        # Merge the DataFrames
        perm_importance_df = pd.merge(
            perm_importance_train_df, 
            perm_importance_test_df, 
            on='Feature'
        )
        
        # Sort by test importance (absolute value)
        perm_importance_df['Abs_Test_Importance'] = perm_importance_df['Test_Importance_Mean'].abs()
        perm_importance_df = perm_importance_df.sort_values('Abs_Test_Importance', ascending=False)
        
        if verbose >= 1:
            print("\nPermutation Feature Importance (Top 10):")
            for i, row in perm_importance_df.head(10).iterrows():
                print(f"  - {row['Feature']}: {row['Test_Importance_Mean']:.6f} ± {row['Test_Importance_Std']:.6f}")
    
    # Create result plots
    if plot_results:
        if verbose >= 1:
            print("\nCreating result plots...")
        
        # Predictions vs Actual plot using Plotly
        fig = go.Figure()

        # Add   
        fig.add_trace(go.Scatter(
            x=df_results['index'].tolist() + df_results['index'].tolist()[::-1],
            y=df_results['upper_bound'].tolist() + df_results['lower_bound'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,176,246,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{confidence_interval*100}% Confidence Interval'
        ))
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=df_results['index'], 
            y=df_results['actual'], 
            mode='lines+markers',
            name='Actual',
            marker=dict(size=6, color='blue'),
            line=dict(width=2)
        ))
        
        # Add predicted values
        fig.add_trace(go.Scatter(
            x=df_results['index'], 
            y=df_results['predicted'], 
            mode='lines+markers',
            name='Predicted',
            marker=dict(size=6, color='red'),
            line=dict(width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Index',
            yaxis_title=target,
            legend=dict(x=0.01, y=0.99),
            height=600,
            width=1000
        )
        
        fig.show()
    
    # Create diagnostic plots
    if plot_diagnostics:
        if verbose >= 1:
            print("\nCreating diagnostic plots...")
        
        # Adjust number of rows based on whether we have permutation importance
        if permutation_importance:
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(df_results['actual'], df_results['predicted'], alpha=0.6)
        
        # Add 45-degree line
        min_val = min(df_results['actual'].min(), df_results['predicted'].min())
        max_val = max(df_results['actual'].max(), df_results['predicted'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted Values (Test Set)')
        
        # Plot 2: Residuals distribution
        residuals = df_results['actual'] - df_results['predicted']
        
        sns.histplot(residuals, kde=True, ax=axes[0, 1], color='blue', alpha=0.6)
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Residuals (Actual - Predicted)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        
        # Plot 3: Feature importance (coefficients)
        if coef_df is not None:
            top_features = coef_df.head(10)
            sns.barplot(x='Coefficient', y='Feature', data=top_features, ax=axes[1, 0])
            axes[1, 0].set_title('Top 10 Features by Coefficient Magnitude')
        else:
            axes[1, 0].text(0.5, 0.5, 'No coefficient data available', 
                            horizontalalignment='center', verticalalignment='center')
        
        # Plot 4: Residuals vs Predicted
        axes[1, 1].scatter(df_results['predicted'], residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Predicted Values')
        
        # Add Permutation Importance Plot if calculated
        if permutation_importance and perm_importance_df is not None:
            # Plot 5: Permutation Importance (Train)
            top_perm_features = perm_importance_df.head(10)
            sns.barplot(
                x='Train_Importance_Mean', 
                y='Feature', 
                data=top_perm_features, 
                ax=axes[2, 0],
                color='skyblue'
            )
            axes[2, 0].set_title('Top 10 Features by Permutation Importance (Training Set)')
            axes[2, 0].set_xlabel('Decrease in Performance')
            
            # Plot 6: Permutation Importance (Test)
            sns.barplot(
                x='Test_Importance_Mean', 
                y='Feature', 
                data=top_perm_features, 
                ax=axes[2, 1],
                color='lightgreen'
            )
            axes[2, 1].set_title('Top 10 Features by Permutation Importance (Test Set)')
            axes[2, 1].set_xlabel('Decrease in Performance')
        
        plt.tight_layout()
        plt.show()
        
        # QQ plot for normality check of residuals
        plt.figure(figsize=(8, 8))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.tight_layout()
        plt.show()

        # # Create a plot showing confidence interval coverage
        # plt.figure(figsize=(12, 6))
        
        # # Sort results by actual values for better visualization
        # sorted_results = df_results.sort_values('actual')
        
        # # Plot actual values
        # plt.plot(range(len(sorted_results)), sorted_results['actual'], 'b-', label='Actual')
        
        # # Plot predicted values
        # plt.plot(range(len(sorted_results)), sorted_results['predicted'], 'r--', label='Predicted')
        
        # # Plot confidence intervals
        # plt.fill_between(
        #     range(len(sorted_results)),
        #     sorted_results['lower_bound'],
        #     sorted_results['upper_bound'],
        #     color='gray', alpha=0.3,
        #     label=f'{confidence_interval*100}% Confidence Interval'
        # )
        
        # # Highlight points outside the confidence interval
        # outside_interval = ~within_interval
        # outside_interval_indices = np.where(outside_interval.values)[0]
        # if len(outside_interval_indices) > 0:
        #     outside_results = sorted_results.iloc[outside_interval_indices]
        #     plt.scatter(
        #         outside_interval_indices,
        #         outside_results['actual'],
        #         color='red', s=50, zorder=5,
        #         label='Outside Confidence Interval'
            # )
        
        # plt.title(f'Predictions with {confidence_interval*100}% Confidence Intervals')
        # plt.xlabel('Sorted Sample Index')
        # plt.ylabel(target)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()


    # Record execution time
    execution_time = time.time() - start_time
    if verbose >= 1:
        print(f"\nExecution completed in {execution_time:.2f} seconds")
    
    # Prepare output series
    pred_test_series = pd.Series(df_results['predicted'].values, index=df_results['index'])
    
    # Create diagnostics dictionary
    diagnostics = {
        'model_type': type(model).__name__,
        'coef_df': coef_df,
        'permutation_importance_df': perm_importance_df,  # Add permutation importance to diagnostics
        'cv_results': cv_results,
        'scaled_features': True,
        'metrics': {
            'mae_train': mae_train,
            'rmse_train': rmse_train,
            'r2_train': r2_train,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test,
            'overfitting_ratio_mae': overfitting_ratio_mae,
            'overfitting_ratio_rmse': overfitting_ratio_rmse
        },
        'execution_time': execution_time,
        # 'feature_scaler': scaler
    }
    
    return model, pred_test_series, mae_train, r2_train, mae_test, r2_test, diagnostics, coef_df, df_results, df_results_train, features_, df_results_reporting




def predict_with_linear_model(
    model, 
    new_data: pd.DataFrame,
    features: list[str],
    confidence_interval: float = 0.95,  # Added confidence interval parameter
    scaler=None
) -> pd.Series:
    """
    Make predictions using a trained linear model
    
    Parameters:
    -----------
    model : 
        Trained model (LinearRegression, Ridge, Lasso, or ElasticNet)
    new_data : pd.DataFrame
        New data to make predictions on  
    features : list[str]
        List of feature names
    confidence_interval : float, default=0.95
        Confidence level for prediction intervals (0-1)
    scaler : 
        Fitted scaler object used during training (optional)
        
    Returns:
    --------
    pd.Series
        Predictions
    """
    # Extract features
    X = new_data[features]
    
    # Scale features if scaler is provided
    if scaler is not None:
        X = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Ensure no negative predictions for positive targets
    predictions = np.maximum(0, predictions)
    
    # Create predictions series
    predictions_series = pd.Series(predictions, index=new_data.index)
    
    # If residual_std is provided, calculate confidence intervals
    if residual_std is not None:
        # Calculate critical value for the desired confidence level
        from scipy import stats
        alpha = 1 - confidence_interval
        # Assuming degrees of freedom from original training
        # This is an approximation - ideally we'd use the actual df from training
        critical_value = stats.t.ppf(1 - alpha/2, df=len(features) * 10)
        
        # Calculate margin of error
        margin_of_error = critical_value * residual_std
        
        # Calculate lower and upper bounds
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        # Ensure lower bound is non-negative for strictly positive targets
        lower_bound = np.maximum(0, lower_bound)
        
        # Create Series objects
        lower_bound_series = pd.Series(lower_bound, index=new_data.index)
        upper_bound_series = pd.Series(upper_bound, index=new_data.index)
        
        return predictions_series, lower_bound_series, upper_bound_series
    
    # Return only predictions if no confidence interval is requested
    return predictions_series

# ================================================================  
# Random Forest
# ================================================================
def random_forest_model(
    df_training: pd.DataFrame, 
    df_test: pd.DataFrame, 
    target: str, 
    features: List[str],
    n_iter: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
    save_model: bool = False,
    model_path: str = 'random_forest_model.pkl',
    plot_results: bool = True,
    handle_zeros: bool = True,
    prevent_overfitting: bool = True
) -> Tuple[Any, np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    Train and evaluate an improved Random Forest regression model with 
    zero-handling and overfitting prevention.
    
    Parameters:
    -----------
    df_training : pd.DataFrame
        Training dataset
    df_test : pd.DataFrame
        Test dataset
    target : str
        Target variable name
    features : List[str]
        List of feature names
    n_iter : int, optional (default=50)
        Number of parameter combinations to try in RandomizedSearchCV
    cv_folds : int, optional (default=5)
        Number of cross-validation folds
    random_state : int, optional (default=42)
        Random seed for reproducibility
    save_model : bool, optional (default=False)
        Whether to save the model to disk
    model_path : str, optional (default='random_forest_model.pkl')
        Path to save the model if save_model is True
    plot_results : bool, optional (default=True)
        Whether to plot results and feature importance
    handle_zeros : bool, optional (default=True)
        Whether to handle the zero-inflation issue
    prevent_overfitting : bool, optional (default=True)
        Whether to apply stricter regularization to prevent overfitting
        
    Returns:
    --------
    Tuple containing:
    - Trained best model
    - Test predictions
    - Dictionary of evaluation metrics
    - Dictionary of additional information (feature importance, etc.)
    """
    print("\n=== Random Forest Model with Zero-Inflation Handling and Overfitting Prevention ===")
    start_time = time.time()
    
    # Create copies to avoid modifying original data
    train_data = df_training.copy()
    test_data = df_test.copy()
    
    # Prepare data
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Analyze zeros in target variable
    zero_count = (y_train == 0).sum()
    zero_percentage = zero_count / len(y_train) * 100
    print(f"\nZero values in target: {zero_count} ({zero_percentage:.2f}%)")
    
    # Handle zero-inflation if requested and if zeros are present
    zero_model = None
    non_zero_indices = None
    if handle_zeros and zero_percentage > 5:
        print("Implementing two-stage modeling approach for zero-inflation:")
        
        # First stage: Predict if value will be zero or non-zero
        y_binary = (y_train > 0).astype(int)
        
        from sklearn.ensemble import RandomForestClassifier
        zero_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced'
        )
        
        # Create validation set for evaluating zero classifier
        X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
            X_train, y_binary, test_size=0.2, random_state=random_state, 
            stratify=y_binary
        )
        
        # Train the classifier
        zero_classifier.fit(X_train_clf, y_train_clf)
        
        # Evaluate zero classifier
        from sklearn.metrics import classification_report
        y_val_pred_clf = zero_classifier.predict(X_val_clf)
        print("\nZero Classifier Performance:")
        print(classification_report(y_val_clf, y_val_pred_clf))
        
        # Save for later use
        zero_model = zero_classifier
        
        # Get non-zero indices for training the regression model
        non_zero_indices = y_train > 0
        print(f"Training regression model on {non_zero_indices.sum()} non-zero samples")
        
        # Update training data for regression model
        X_train_reg = X_train[non_zero_indices]
        y_train_reg = y_train[non_zero_indices]
    else:
        # If not handling zeros separately, use all data
        X_train_reg = X_train
        y_train_reg = y_train
    
    # Define parameter grid based on overfitting prevention
    if prevent_overfitting:
        print("\nUsing stricter regularization to prevent overfitting...")
        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [5, 10, 15, 20],  # Reduced max depth
            'rf__min_samples_split': [5, 10, 15],  # Increased min samples
            'rf__min_samples_leaf': [2, 4, 8],     # Increased min samples leaf
            'rf__max_features': ['sqrt', 'log2'],  # Restrict feature selection
            'rf__bootstrap': [True],
            'rf__max_samples': [0.7, 0.8],         # Use subsampling
            'rf__criterion': ['squared_error', 'absolute_error']
        }
    else:
        param_grid = {
            'rf__n_estimators': [100, 200, 300, 400],
            'rf__max_depth': [10, 20, 30, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['sqrt', 'log2', None],
            'rf__bootstrap': [True, False],
            'rf__max_samples': [0.7, 0.8, 0.9, None],
            'rf__criterion': ['squared_error', 'absolute_error', 'poisson']
        }
    
    print(f"\nStarting hyperparameter tuning with {n_iter} iterations and {cv_folds}-fold CV...")
    
    # Create pipeline with feature scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(random_state=random_state, n_jobs=-1))
    ])
    
    # Randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(
        estimator=pipeline, 
        param_distributions=param_grid,
        n_iter=n_iter, 
        cv=cv_folds, 
        verbose=1, 
        n_jobs=-1,
        scoring='neg_mean_absolute_error',
        return_train_score=True,
        random_state=random_state
    )
    
    # Fit on appropriate training data (either all data or non-zero only)
    random_search.fit(X_train_reg, y_train_reg)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print("\nBest parameters found:")
    for param, value in best_params.items():
        print(f"  - {param}: {value}")
    
    # Cross-validation for more robust evaluation
    print("\nPerforming cross-validation...")
    cv_scores = cross_val_score(
        best_model, X_train_reg, y_train_reg, 
        cv=cv_folds, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    print(f"CV MAE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
    
    # Get predictions
    if handle_zeros and zero_percentage > 5:
        # First predict zero vs non-zero
        zero_predictions_train = zero_classifier.predict(X_train)
        zero_predictions_test = zero_classifier.predict(X_test)
        
        # Then predict values for non-zero instances
        y_pred_train_raw = np.zeros(len(X_train))
        non_zero_pred_train = best_model.predict(X_train[zero_predictions_train == 1])
        y_pred_train_raw[zero_predictions_train == 1] = non_zero_pred_train
        
        y_pred_test_raw = np.zeros(len(X_test))
        non_zero_pred_test = best_model.predict(X_test[zero_predictions_test == 1])
        y_pred_test_raw[zero_predictions_test == 1] = non_zero_pred_test
        
        # Ensure no negative predictions
        y_pred_train = np.maximum(0, y_pred_train_raw)
        y_pred_test = np.maximum(0, y_pred_test_raw)
    else:
        # Standard prediction
        y_pred_train_raw = best_model.predict(X_train)
        y_pred_test_raw = best_model.predict(X_test)
        
        # Ensure no negative predictions
        y_pred_train = np.maximum(0, y_pred_train_raw)
        y_pred_test = np.maximum(0, y_pred_test_raw)
    
    # Evaluate model performance
    metrics = {}
    metrics['mae_train'] = mean_absolute_error(y_train, y_pred_train)
    metrics['rmse_train'] = np.sqrt(mean_squared_error(y_train, y_pred_train))
    metrics['r2_train'] = r2_score(y_train, y_pred_train)
    metrics['mae_test'] = mean_absolute_error(y_test, y_pred_test)
    metrics['rmse_test'] = np.sqrt(mean_squared_error(y_test, y_pred_test))
    metrics['r2_test'] = r2_score(y_test, y_pred_test)
    metrics['cv_mae'] = -cv_scores.mean()
    metrics['cv_mae_std'] = cv_scores.std()
    
    # Calculate overfitting ratio (higher values indicate more overfitting)
    metrics['overfitting_ratio_mae'] = metrics['mae_test'] / metrics['mae_train']
    metrics['overfitting_ratio_rmse'] = metrics['rmse_test'] / metrics['rmse_train']
    
    print("\nModel Performance:")
    print(f"  Training Set:")
    print(f"    - MAE: {metrics['mae_train']:.2f}")
    print(f"    - RMSE: {metrics['rmse_train']:.2f}")
    print(f"    - R²: {metrics['r2_train']:.2f}")
    print(f"  Test Set:")
    print(f"    - MAE: {metrics['mae_test']:.2f}")
    print(f"    - RMSE: {metrics['rmse_test']:.2f}")
    print(f"    - R²: {metrics['r2_test']:.2f}")
    print(f"  Overfitting Assessment:")
    print(f"    - MAE Ratio (Test/Train): {metrics['overfitting_ratio_mae']:.2f} (>1.3 indicates overfitting)")
    print(f"    - RMSE Ratio (Test/Train): {metrics['overfitting_ratio_rmse']:.2f} (>1.3 indicates overfitting)")
    
    # Calculate feature importance
    print("\nCalculating feature importance...")
    
    # Default feature importance (from the Random Forest regressor inside the pipeline)
    rf_model = best_model.named_steps['rf']
    default_importance = pd.Series(
        rf_model.feature_importances_, 
        index=features
    ).sort_values(ascending=False)
    
    # Permutation importance (more reliable than default)
    perm_importance = permutation_importance(
        best_model, X_test, y_test, 
        n_repeats=10, 
        random_state=random_state, 
        n_jobs=-1
    )
    
    perm_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 features by permutation importance:")
    for i, row in perm_importance_df.head(10).iterrows():
        print(f"  - {row['Feature']}: {row['Importance']:.4f} (±{row['Std']:.4f})")
    
    # Error analysis
    train_residuals = y_train - y_pred_train
    test_residuals = y_test - y_pred_test
    
    # Special analysis for zero values
    zero_indices_test = np.where(y_test == 0)[0]
    if len(zero_indices_test) > 0:
        zero_pred_values = y_pred_test[zero_indices_test]
        print(f"\nPerformance on zero values (n={len(zero_indices_test)}):")
        print(f"  - Mean predicted value: {zero_pred_values.mean():.4f}")
        print(f"  - Max predicted value: {zero_pred_values.max():.4f}")
        print(f"  - Correctly predicted as zero: {(zero_pred_values < 0.01).sum()} ({(zero_pred_values < 0.01).sum() / len(zero_indices_test) * 100:.1f}%)")
    
    # Create visualization if requested
    if plot_results:
        print("\nGenerating plots...")
        fig, axs = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Actual vs Predicted
        axs[0, 0].scatter(y_test, y_pred_test, alpha=0.5)
        axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axs[0, 0].set_xlabel('Actual Values')
        axs[0, 0].set_ylabel('Predicted Values')
        axs[0, 0].set_title('Actual vs Predicted Values (Test Set)')
        
        # Highlight zero actual values
        if len(zero_indices_test) > 0:
            axs[0, 0].scatter(
                y_test[zero_indices_test], 
                y_pred_test[zero_indices_test], 
                color='red', 
                alpha=0.7,
                label=f'Zero actual values (n={len(zero_indices_test)})'
            )
            axs[0, 0].legend()
        
        # Plot 2: Residuals Distribution
        axs[0, 1].hist(test_residuals, bins=30, alpha=0.7, color='blue', label='All residuals')
        
        # Distribution of residuals for zero actual values
        if len(zero_indices_test) > 0:
            zero_residuals = test_residuals[zero_indices_test]
            axs[0, 1].hist(zero_residuals, bins=15, alpha=0.5, color='red', label='Zero value residuals')
            axs[0, 1].legend()
        
        axs[0, 1].axvline(x=0, color='black', linestyle='--')
        axs[0, 1].set_xlabel('Residual Values (Actual - Predicted)')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].set_title('Distribution of Residuals (Test Set)')
        
        # Plot 3: Top 10 Feature Importance
        top_features = perm_importance_df.head(10)
        axs[1, 0].barh(top_features['Feature'], top_features['Importance'])
        axs[1, 0].set_xlabel('Importance')
        axs[1, 0].set_title('Top 10 Features by Permutation Importance')
        
        # Plot 4: Training vs Test Error (Learning Curves visualization)
        param_values = list(random_search.cv_results_['param_rf__n_estimators'].data)
        train_scores = []
        test_scores = []
        param_map = {}
        
        for i, param in enumerate(param_values):
            if param is not None:
                value = param
                if value not in param_map:
                    param_map[value] = {
                        'train_scores': [],
                        'test_scores': []
                    }
                
                train_score = -random_search.cv_results_['mean_train_score'][i]
                test_score = -random_search.cv_results_['mean_test_score'][i]
                
                param_map[value]['train_scores'].append(train_score)
                param_map[value]['test_scores'].append(test_score)
        
        # Calculate mean scores for each parameter value
        param_values = sorted(param_map.keys())
        train_means = [np.mean(param_map[p]['train_scores']) for p in param_values]
        test_means = [np.mean(param_map[p]['test_scores']) for p in param_values]
        
        # Plot the learning curves
        axs[1, 1].plot(param_values, train_means, 'o-', color='blue', label='Training MAE')
        axs[1, 1].plot(param_values, test_means, 'o-', color='red', label='Test MAE')
        axs[1, 1].set_xlabel('Number of Estimators')
        axs[1, 1].set_ylabel('Mean Absolute Error')
        axs[1, 1].set_title('Training vs Test Error')
        axs[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Additional plot specifically for the zero-inflation issue
        if len(zero_indices_test) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create a scatter plot with color coding
            zero_actual = np.zeros(len(y_test), dtype=bool)
            zero_actual[zero_indices_test] = True
            
            # Non-zero actual values
            ax.scatter(
                np.arange(len(y_test))[~zero_actual], 
                y_pred_test[~zero_actual], 
                color='blue', 
                alpha=0.5,
                label='Non-zero actual values'
            )
            
            # Zero actual values
            ax.scatter(
                np.arange(len(y_test))[zero_actual], 
                y_pred_test[zero_actual], 
                color='red', 
                alpha=0.7,
                label='Zero actual values'
            )
            
            ax.axhline(y=0, color='black', linestyle='--')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Predicted Value')
            ax.set_title('Predicted Values by Actual Value Type')
            ax.legend()
            
            plt.tight_layout()
            plt.show()
    
    # Save model if requested
    if save_model:
        print(f"\nSaving model to {model_path}")
        model_to_save = {
            'main_model': best_model,
            'zero_classifier': zero_model if handle_zeros and zero_percentage > 5 else None,
            'features': features,
            'target': target,
            'handle_zeros': handle_zeros and zero_percentage > 5,
            'metrics': metrics
        }
        joblib.dump(model_to_save, model_path)
    
    # Record execution time
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    
    # Prepare additional information to return
    additional_info = {
        'feature_importance': perm_importance_df,
        'best_params': best_params,
        'cv_results': random_search.cv_results_,
        'train_residuals': train_residuals,
        'test_residuals': test_residuals,
        'zero_analysis': {
            'zero_percentage': zero_percentage,
            'zero_indices_test': zero_indices_test,
            'zero_predictions': y_pred_test[zero_indices_test] if len(zero_indices_test) > 0 else None
        },
        'execution_time': execution_time,
        'zero_model': zero_model if handle_zeros and zero_percentage > 5 else None
    }
    
    return best_model, y_pred_test, metrics, additional_info

def predict_with_saved_model(model_path, new_data):
    """
    Make predictions using a saved model
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    new_data : pd.DataFrame
        New data to make predictions on
        
    Returns:
    --------
    np.ndarray
        Predictions
    """
    # Load the model
    model_dict = joblib.load(model_path)
    
    # Extract components
    main_model = model_dict['main_model']
    zero_classifier = model_dict['zero_classifier']
    features = model_dict['features']
    handle_zeros = model_dict['handle_zeros']
    
    # Prepare features
    X = new_data[features]
    
    # Make predictions
    if handle_zeros and zero_classifier is not None:
        # First predict zero vs non-zero
        zero_predictions = zero_classifier.predict(X)
        
        # Then predict values for non-zero instances
        y_pred_raw = np.zeros(len(X))
        if any(zero_predictions == 1):
            non_zero_pred = main_model.predict(X[zero_predictions == 1])
            y_pred_raw[zero_predictions == 1] = non_zero_pred
    else:
        # Standard prediction
        y_pred_raw = main_model.predict(X)
    
    # Ensure no negative predictions
    y_pred = np.maximum(0, y_pred_raw)
    
    return y_pred


# ================================================================
# LightGBM model
# ================================================================
# LightGBM is a gradient boosting framework that uses tree based learning algorithms
def lightgbm_model(
    df_training: pd.DataFrame, 
    df_testing: pd.DataFrame, 
    name_time_column: str, 
    target: str, 
    features: List[str],
    optimization_method: str = 'optuna',  # 'optuna', 'randomsearch', or 'cv'
    n_trials: int = 50,
    use_zero_inflated: bool = False,
    handle_outliers: bool = True,
    n_folds: int = 5,
    time_series_cv: bool = True,
    plot_results: bool = True,
    save_model: bool = False,
    model_path: str = 'lightgbm_model.txt',
    verbose: int = 1,
    random_state: int = 42
) -> Tuple[Dict, pd.Series, float, float, float, float, Dict]:
    """
    Enhanced LightGBM model for time series regression with advanced hyperparameter tuning, 
    feature importance analysis, and diagnostics.
    
    Parameters:
    -----------
    df_training : pd.DataFrame
        Training dataframe with time series data
    df_testing : pd.DataFrame
        Testing dataframe with time series data
    name_time_column : str
        Name of the time column in both dataframes
    target : str
        Target variable name
    features : List[str]
        List of feature column names
    optimization_method : str, optional (default='optuna')
        Method for hyperparameter optimization: 'optuna', 'randomsearch', or 'cv'
    n_trials : int, optional (default=50)
        Number of optimization trials for Optuna or RandomizedSearchCV
    use_zero_inflated : bool, optional (default=False)
        Whether to use a two-stage model for zero-inflated target
    handle_outliers : bool, optional (default=True)
        Whether to apply special treatment for outliers
    n_folds : int, optional (default=5)
        Number of folds for cross-validation
    time_series_cv : bool, optional (default=True)
        Whether to use time series cross-validation instead of random CV
    plot_results : bool, optional (default=True)
        Whether to generate diagnostic plots
    save_model : bool, optional (default=False)
        Whether to save the model to disk
    model_path : str, optional (default='lightgbm_model.txt')
        Path to save the model if save_model is True
    verbose : int, optional (default=1)
        Verbosity level (0: silent, 1: important info, 2: detailed)
    random_state : int, optional (default=42)
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple containing:
    - Model information dictionary
    - Test predictions Series
    - Training MAE
    - Training R²
    - Test MAE
    - Test R²
    - Additional diagnostics dictionary
    """
    start_time = time.time()
    if verbose >= 1:
        print("\n" + "="*80)
        print(" Enhanced LightGBM Model for Time Series Regression ")
        print("="*80)
    
    # Create copies to avoid modifying original data
    train_data = df_training.copy().reset_index(drop=True)
    test_data = df_testing.copy().reset_index(drop=True)
    
    # Ensure time column is datetime
    for df in [train_data, test_data]:
        if name_time_column in df.columns and not pd.api.types.is_datetime64_any_dtype(df[name_time_column]):
            try:
                df[name_time_column] = pd.to_datetime(df[name_time_column])
            except Exception as e:
                if verbose >= 1:
                    print(f"Warning: Could not convert {name_time_column} to datetime. Error: {str(e)}")
    
    # Sort by time
    if name_time_column in train_data.columns:
        train_data = train_data.sort_values(by=name_time_column)
        test_data = test_data.sort_values(by=name_time_column)
    
    # Analyze target distribution
    zero_percentage = (train_data[target] == 0).mean() * 100
    if verbose >= 1:
        print(f"\nTarget analysis:")
        print(f"- Range: [{train_data[target].min()}, {train_data[target].max()}]")
        print(f"- Mean: {train_data[target].mean():.4f}")
        print(f"- Zeros: {zero_percentage:.2f}%")
    
    # Handle zero inflation if requested
    zero_classifier = None
    if use_zero_inflated and zero_percentage > 5:
        if verbose >= 1:
            print("\nHandling zero-inflation with two-stage model...")
        
        # Create binary target (zero vs non-zero)
        train_data['is_zero'] = (train_data[target] == 0).astype(int)
        
        # Train zero classifier
        zero_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
        
        zero_train_data = lgb.Dataset(train_data[features], train_data['is_zero'])
        zero_classifier = lgb.train(
            params=zero_params,
            train_set=zero_train_data,
            num_boost_round=100,
            verbose_eval=False
        )
        
        # Keep only non-zero records for regression model
        non_zero_mask = (train_data[target] > 0)
        train_data_reg = train_data[non_zero_mask].copy()
        
        if verbose >= 1:
            print(f"- Using {len(train_data_reg)} non-zero samples for regression model")
    else:
        train_data_reg = train_data.copy()
    
    # Handle outliers if requested
    if handle_outliers:
        q1 = train_data_reg[target].quantile(0.25)
        q3 = train_data_reg[target].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        outliers = train_data_reg[target] > upper_bound
        outlier_count = outliers.sum()
        
        if outlier_count > 0 and outlier_count < len(train_data_reg) * 0.1:  # Don't remove too many
            if verbose >= 1:
                print(f"\nHandling {outlier_count} outliers ({outlier_count/len(train_data_reg)*100:.2f}%)")
            
            # Use weighted samples instead of removing
            train_data_reg['weight'] = 1.0
            train_data_reg.loc[outliers, 'weight'] = 0.3  # Lower weight for outliers
        else:
            train_data_reg['weight'] = 1.0
    else:
        train_data_reg['weight'] = 1.0
    
    # Prepare datasets
    X_train = train_data_reg[features]
    y_train = train_data_reg[target]
    weights = train_data_reg['weight']
    
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Cross-validation strategy
    if time_series_cv:
        cv = TimeSeriesSplit(n_splits=n_folds)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Feature engineering - time-based features
    if name_time_column in train_data.columns:
        for df, X in [(train_data_reg, X_train), (test_data, X_test)]:
            if 'hour' not in features and 'dayofweek' not in features:
                try:
                    # Add hour and day of week if not already in features
                    X['hour'] = df[name_time_column].dt.hour
                    X['dayofweek'] = df[name_time_column].dt.dayofweek
                    features.extend(['hour', 'dayofweek'])
                    if verbose >= 1:
                        print("\nAdded time features: hour, dayofweek")
                except:
                    if verbose >= 1:
                        print("Could not add time features")
    
    # Create LightGBM datasets
    lgb_train = lgb.Dataset(X_train, y_train, weight=weights, free_raw_data=False)
    
    # Model optimization
    best_params = {}
    
    if optimization_method == 'optuna':
        if verbose >= 1:
            print("\nOptimizing hyperparameters with Optuna...")
        
        def objective(trial):
            param = {
                'objective': 'regression',
                'metric': 'mae',
                'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 0.5, log=True),
                'num_iterations': 10,#trial.suggest_int('num_iterations', 100, 1000),
                'feature_pre_filter': False,  # Add this line
                'verbose': -1,
                'random_state': random_state
            }
            callbacks = [
                lgb.early_stopping(stopping_rounds=3),
                lgb.log_evaluation(period=50)  # stampa ogni 50 iterazioni
            ]
            # Use cross-validation for evaluation
            cv_results = lgb.cv(
                params=param,
                train_set=lgb_train,
                folds=cv,
                num_boost_round=param['num_iterations'],
                callbacks=callbacks,
                shuffle=False,
                metrics=['mae']
            )
            
            # Get the best iteration
            # return print(cv_results.keys())
            best_score = min(cv_results['valid l1-mean'])
            return best_score
        
        # Create and run Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_params['random_state'] = random_state
        best_params['objective'] = 'regression'
        best_params['metric'] = 'mae'
        best_params['verbose'] = -1
        
        # Get number of iterations
        num_iterations = best_params.pop('num_iterations', 1000)
        
    elif optimization_method == 'randomsearch':
        if verbose >= 1:
            print("\nOptimizing hyperparameters with RandomizedSearchCV...")
        
        param_dist = {
            'num_leaves': list(range(20, 151, 10)),
            'learning_rate': list(np.logspace(-2, -0.5, 20)),
            'feature_fraction': list(np.linspace(0.5, 1.0, 11)),
            'bagging_fraction': list(np.linspace(0.5, 1.0, 11)),
            'bagging_freq': list(range(1, 11)),
            'min_child_samples': list(range(5, 101, 5)),
            'reg_alpha': list(np.logspace(-8, 1, 10)),
            'reg_lambda': list(np.logspace(-8, 1, 10)),
            'min_split_gain': list(np.logspace(-8, -1, 8))
        }
        
        lgbm = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
        
        random_search = RandomizedSearchCV(
            estimator=lgbm,
            param_distributions=param_dist,
            n_iter=n_trials,
            cv=cv,
            scoring='neg_mean_absolute_error',
            random_state=random_state,
            n_jobs=-1,
            verbose=0 if verbose < 2 else 1
        )
        
        random_search.fit(X_train, y_train, sample_weight=weights)
        
        # Get best parameters
        best_params = random_search.best_estimator_.get_params()
        best_params = {k: v for k, v in best_params.items() if k in param_dist}
        best_params['random_state'] = random_state
        best_params['objective'] = 'regression'
        best_params['metric'] = 'mae'
        best_params['verbose'] = -1
        
        num_iterations = 1000  # Default value
        
    else:  # Use default CV approach
        if verbose >= 1:
            print("\nRunning default LightGBM cross-validation...")
        
        best_params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state
        }
        
        # Run cross-validation
        cv_results = lgb.cv(
            params=best_params,
            train_set=lgb_train,
            folds=cv,
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Get the best number of iterations
        num_iterations = len(cv_results['valid mae-mean'])
    
    if verbose >= 1:
        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"  - {param}: {value}")
    
    # Train final model
    if verbose >= 1:
        print(f"\nTraining final model with {num_iterations} iterations...")
    
    final_model = lgb.train(
        params=best_params,
        train_set=lgb_train,
        num_boost_round=num_iterations
    )
    
    # Make predictions
    if use_zero_inflated and zero_percentage > 5 and zero_classifier is not None:
        # For training data
        zero_pred_train = zero_classifier.predict(train_data[features]) > 0.5
        
        y_pred_train = np.zeros(len(train_data))
        if np.any(~zero_pred_train):
            y_pred_train[~zero_pred_train] = 0
        
        if np.any(zero_pred_train):
            non_zero_pred_train = final_model.predict(train_data.loc[zero_pred_train, features])
            y_pred_train[zero_pred_train] = non_zero_pred_train
        
        # For test data
        zero_pred_test = zero_classifier.predict(test_data[features]) > 0.5
        
        y_pred_test = np.zeros(len(test_data))
        if np.any(~zero_pred_test):
            y_pred_test[~zero_pred_test] = 0
        
        if np.any(zero_pred_test):
            non_zero_pred_test = final_model.predict(test_data.loc[zero_pred_test, features])
            y_pred_test[zero_pred_test] = non_zero_pred_test
    else:
        # Standard prediction
        y_pred_train = final_model.predict(train_data[features])
        y_pred_test = final_model.predict(test_data[features])
    
    # Convert to pandas Series
    pred_train_series = pd.Series(y_pred_train, index=train_data.index)
    pred_test_series = pd.Series(y_pred_test, index=test_data.index)
    
    # Calculate metrics
    mae_train = mean_absolute_error(train_data[target], y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(train_data[target], y_pred_train))
    r2_train = r2_score(train_data[target], y_pred_train)
    
    mae_test = mean_absolute_error(test_data[target], y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(test_data[target], y_pred_test))
    r2_test = r2_score(test_data[target], y_pred_test)
    
    # Overfitting metrics
    overfitting_ratio_mae = mae_test / mae_train if mae_train > 0 else float('inf')
    overfitting_ratio_rmse = rmse_test / rmse_train if rmse_train > 0 else float('inf')
    
    if verbose >= 1:
        print("\nModel Performance:")
        print(f"  Training Set:")
        print(f"    - MAE: {mae_train:.4f}")
        print(f"    - RMSE: {rmse_train:.4f}")
        print(f"    - R²: {r2_train:.4f}")
        print(f"  Test Set:")
        print(f"    - MAE: {mae_test:.4f}")
        print(f"    - RMSE: {rmse_test:.4f}")
        print(f"    - R²: {r2_test:.4f}")
        print(f"  Overfitting Assessment:")
        print(f"    - MAE Ratio (Test/Train): {overfitting_ratio_mae:.2f} (>1.3 indicates overfitting)")
        print(f"    - RMSE Ratio (Test/Train): {overfitting_ratio_rmse:.2f} (>1.3 indicates overfitting)")
    
    # Calculate feature importance
    if verbose >= 1:
        print("\nFeature Importance:")
    
    importance = final_model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    if verbose >= 1:
        for i, row in importance_df.head(10).iterrows():
            print(f"  - {row['Feature']}: {row['Importance']:.2f}")
    
    # Special analysis for zero values
    zero_indices_test = np.where(test_data[target] == 0)[0]
    zero_analysis = {}
    
    if len(zero_indices_test) > 0:
        zero_pred_values = y_pred_test[zero_indices_test]
        zero_analysis = {
            'count': len(zero_indices_test),
            'mean_pred': np.mean(zero_pred_values),
            'max_pred': np.max(zero_pred_values),
            'correct_zeros': np.sum(zero_pred_values < 0.01),
            'accuracy': np.sum(zero_pred_values < 0.01) / len(zero_indices_test)
        }
        
        if verbose >= 1:
            print(f"\nPerformance on zero values (n={len(zero_indices_test)}):")
            print(f"  - Mean predicted value: {zero_analysis['mean_pred']:.4f}")
            print(f"  - Correctly predicted as zero: {zero_analysis['correct_zeros']} ({zero_analysis['accuracy']*100:.1f}%)")
    
    # Create diagnostic plots
    if plot_results:
        if verbose >= 1:
            print("\nCreating diagnostic plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Actual vs Predicted (with zero values highlighted)
        axes[0, 0].scatter(test_data[target], y_pred_test, alpha=0.5, color='blue')
        if len(zero_indices_test) > 0:
            axes[0, 0].scatter(
                test_data[target].iloc[zero_indices_test],
                y_pred_test[zero_indices_test],
                color='red',
                alpha=0.7,
                label=f'Zero actual values (n={len(zero_indices_test)})'
            )
            axes[0, 0].legend()
        
        # Add 45-degree line
        min_val = min(test_data[target].min(), y_pred_test.min())
        max_val = max(test_data[target].max(), y_pred_test.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].set_title('Actual vs Predicted Values (Test Set)')
        
        # Plot 2: Residuals distribution
        residuals = test_data[target] - y_pred_test
        
        sns.histplot(residuals, kde=True, ax=axes[0, 1], color='blue', alpha=0.6)
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Residuals (Actual - Predicted)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        
        # Plot 3: Feature importance
        top_features = importance_df.head(10)
        sns.barplot(x='Importance', y='Feature', data=top_features, ax=axes[1, 0])
        axes[1, 0].set_title('Top 10 Features by Importance')
        
        # Plot 4: Time series of actual vs predicted
        if name_time_column in test_data.columns:
            test_results = pd.DataFrame({
                'Actual': test_data[target],
                'Predicted': y_pred_test,
                'Time': test_data[name_time_column]
            }).sort_values('Time')
            
            axes[1, 1].plot(test_results['Time'], test_results['Actual'], label='Actual', alpha=0.7)
            axes[1, 1].plot(test_results['Time'], test_results['Predicted'], label='Predicted', alpha=0.7)
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Target Value')
            axes[1, 1].set_title('Time Series of Actual vs Predicted')
            axes[1, 1].legend()
            
            # Rotate date labels for better readability
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
        else:
            # Alternative plot: Residuals vs Predicted
            axes[1, 1].scatter(y_pred_test, residuals, alpha=0.5)
            axes[1, 1].axhline(y=0, color='red', linestyle='--')
            axes[1, 1].set_xlabel('Predicted Values')
            axes[1, 1].set_ylabel('Residuals')
            axes[1, 1].set_title('Residuals vs Predicted Values')
            
        plt.tight_layout()
        plt.show()
        
        # Additional plot specifically for zero values if they exist
        if len(zero_indices_test) > 0:
            plt.figure(figsize=(10, 6))
            
            # Create a scatter plot with color coding
            zero_actual = np.zeros(len(test_data[target]), dtype=bool)
            zero_actual[zero_indices_test] = True
            
            # Non-zero actual values
            plt.scatter(
                np.arange(len(test_data))[~zero_actual],
                y_pred_test[~zero_actual],
                color='blue',
                alpha=0.5,
                label='Non-zero actual values'
            )
            
            # Zero actual values
            plt.scatter(
                np.arange(len(test_data))[zero_actual],
                y_pred_test[zero_actual],
                color='red',
                alpha=0.7,
                label='Zero actual values'
            )
            
            plt.axhline(y=0, color='black', linestyle='--')
            plt.xlabel('Sample Index')
            plt.ylabel('Predicted Value')
            plt.title('Predicted Values by Actual Value Type')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    # Calculate SHAP values for feature interpretation
    explainer = None
    shap_values = None
    
    try:
        if verbose >= 1:
            print("\nCalculating SHAP values for model interpretation...")
        
        # Use a sample of training data for SHAP calculation to improve speed
        max_sample = min(1000, len(X_train))
        X_sample = X_train.sample(max_sample, random_state=random_state)
        
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_sample)
        
        if plot_results:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            plt.show()
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        if verbose >= 1:
            print(f"Warning: SHAP analysis failed. Error: {str(e)}")
    
    # Save model if requested
    if save_model:
        if verbose >= 1:
            print(f"\nSaving model to {model_path}")
        
        final_model.save_model(model_path)
    
    # Record execution time
    execution_time = time.time() - start_time
    if verbose >= 1:
        print(f"\nExecution completed in {execution_time:.2f} seconds")
    
    # Create model info dictionary
    model_info = {
        "data": train_data,
        "features": features,
        "target": target,
        "time_column": name_time_column,
        "params": best_params,
        "estimator": final_model,
        "zero_classifier": zero_classifier if use_zero_inflated and zero_percentage > 5 else None,
        "use_zero_inflated": use_zero_inflated and zero_percentage > 5,
        "metrics": {
            "mae_train": mae_train,
            "rmse_train": rmse_train,
            "r2_train": r2_train,
            "mae_test": mae_test,
            "rmse_test": rmse_test,
            "r2_test": r2_test,
            "overfitting_ratio_mae": overfitting_ratio_mae,
            "overfitting_ratio_rmse": overfitting_ratio_rmse
        }
    }
    
    # Create diagnostics dictionary
    diagnostics = {
        "feature_importance": importance_df,
        "zero_analysis": zero_analysis if len(zero_indices_test) > 0 else None,
        "explainer": explainer,
        "shap_values": shap_values,
        "execution_time": execution_time,
        "optimization_method": optimization_method,
        "n_trials": n_trials
    }
    
    return model_info, pred_test_series, mae_train, r2_train, mae_test, r2_test, diagnostics


def predict_with_lightgbm_model(model_info: Dict, new_data: pd.DataFrame) -> pd.Series:
    """
    Make predictions using a trained LightGBM model
    
    Parameters:
    -----------
    model_info : Dict
        Model information dictionary from improved_lightgbm_model
    new_data : pd.DataFrame
        New data to make predictions on
        
    Returns:
    --------
    pd.Series
        Predictions
    """
    # Extract model components
    features = model_info["features"]
    estimator = model_info["estimator"]
    zero_classifier = model_info.get("zero_classifier")
    use_zero_inflated = model_info.get("use_zero_inflated", False)
    
    # Prepare features
    X = new_data[features]
    
    # Make predictions
    if use_zero_inflated and zero_classifier is not None:
        # First predict zero vs non-zero
        zero_pred = zero_classifier.predict(X) > 0.5
        
        # Initialize predictions with zeros
        y_pred = np.zeros(len(X))
        
        # Predict values for non-zero instances
        if np.any(zero_pred):
            non_zero_pred = estimator.predict(X.loc[zero_pred])
            y_pred[zero_pred] = non_zero_pred
    else:
        # Standard prediction
        y_pred = estimator.predict(X)
    
    # Ensure no negative predictions for positive targets
    y_pred = np.maximum(0, y_pred)
    
    # Return as Series
    return pd.Series(y_pred, index=new_data.index)

# ================================================================
# Neura network
# ================================================================

def create_improved_lstm_model(input_shape, learning_rate=0.001):
    model = Sequential()
    
    model.add(Input(shape=(input_shape)))
    # First LSTM layer with increased units and bidirectional wrapper
    model.add(Bidirectional(LSTM(128, activation='relu', 
                                 return_sequences=True,
                                 kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                                 recurrent_regularizer=l1_l2(l1=0, l2=1e-4),
                                 bias_regularizer=l1_l2(l1=0, l2=1e-4),
                                 dropout=0.2, recurrent_dropout=0.2)))
    
    # LeakyReLU to avoid vanishing gradients
    model.add(LeakyReLU(negative_slope=0.1))
    # Batch normalization helps with training stability
    model.add(BatchNormalization())
    
    # Second LSTM layer
    model.add(LSTM(64, activation='relu',
                   kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                   recurrent_regularizer=l1_l2(l1=0, l2=1e-4),
                   bias_regularizer=l1_l2(l1=0, l2=1e-4),
                   dropout=0.3, recurrent_dropout=0.3))
    
    # LeakyReLU activation after the LSTM layer
    model.add(LeakyReLU(negative_slope=0.1))
    
    # Batch normalization
    model.add(BatchNormalization())
    
    # Dense hidden layer
    model.add(Dense(32, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))
    model.add(LeakyReLU(negative_slope=0.1))
    
    # Dropout
    model.add(Dropout(0.2))
    
    # Output layer (no activation function, as this is regression)
    model.add(Dense(1))
    
    # Use Adam optimizer with the customized learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()


def build_tensorflow_model(df_training: pd.DataFrame, target: str, adjust_prediction: bool, adjusted_value_prediction:float, features: list[str], \
    epochs_model: int, batch_size_model: int):


    # Scale data
    X = df_training[features].reset_index(drop=True)
    y = df_training[target].reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Scale data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Create the model with your input shape
    input_shape = (X_train.shape[1], 1)
    improved_model = create_improved_lstm_model(input_shape)

    # Create callbacks for better training
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        
        # Reduce learning rate when a metric has stopped improving
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        
        # Save the best model during training
        ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Train the model with these improvements
    history = improved_model.fit(
        X_train, y_train,
        # epochs=100,
        epochs=epochs_model,
        # batch_size=32,
        batch_size=batch_size_model,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate the model
    train_loss = improved_model.evaluate(X_train, y_train, verbose=0)
    print(f"Training loss: {train_loss[0]:.4f}, MAE: {train_loss[1]:.4f}")

    # If you have a test set
    test_loss = improved_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss[0]:.4f}, MAE: {test_loss[1]:.4f}")

    # Optionally, visualize the learning process
    plot_history(history)

    # Prediction output
    y_pred_train =improved_model.predict(X_train) 
    y_pred_train = y_pred_train.flatten().tolist()
    
    y_pred_test = improved_model.predict(X_test)
    y_pred_test = y_pred_test.flatten().tolist()


    if adjust_prediction:
        y_pred_modified = []
        for data in y_pred_test:
            if data < adjusted_value_prediction:
                y_pred_modified.append(0)
            else:
                y_pred_modified.append(data)
    else:
        y_pred_modified = y_pred_test

    # Metrics
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test[:2000], y_pred_modified[:2000])
    r2_test = r2_score(y_test[:2000], y_pred_modified[:2000])
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_modified[:2000])
    # Difference between R2 train and test
    deltaR = r2_train - r2_test


    print(f"MAE - train: {mae_train:.2f}, R² - train: {r2_train:.2f}")
    print(f"MAE - test: {mae_test:.2f}, R² - test: {r2_test:.2f}")

    plt.plot(y_test, label='True')
    plt.plot(y_pred_modified, label='Predicted')
    plt.legend()
    plt.title("Time Series Forecasting (LSTM)")
    plt.grid(True)
    plt.show()

    return improved_model,y_pred_modified, mae_train, r2_train, mae_test, r2_test, mse_train, mse_test, deltaR
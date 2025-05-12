import pandas as pd
from functions_analysis import *
from models import *
import report.function_graphs as BlgGraphs

def pre_processing_data(df:pd.DataFrame, time_col_name: str, features_temperature: list[str], features_energy: list[str], resample_data: bool, resample_type: str) -> pd.DataFrame:
    # df[time_col_name] = pd.to_datetime(df[time_col_name], format="%d/%m/%Y %H:%M:%S")
    df[time_col_name] = pd.to_datetime(df[time_col_name])
    df.index = df[time_col_name]
    del df[time_col_name]
    
    # Remove column wit all nan
    df = df.dropna(axis=1, how='all')
    # Drop na values 
    df = df.dropna()

    # Case energy
    df_temperature = df.loc[:, features_temperature]
    df_energy = df.loc[:, features_energy]
    # remove outliers



    if resample_data:
        df_temperature_resample = df_temperature.resample(resample_type).mean()
        df_energy_resample = df_energy.resample(resample_type).sum()
        df_resample = pd.concat([df_temperature_resample, df_energy_resample], axis=1)
        df_resample['hour'] = df_resample.index.hour
        df_resample['dayofweek'] = df_resample.index.dayofweek
        df_resample['week'] = df_resample.index.isocalendar().week.astype('int')
        df_resample =df_resample.dropna()
        # df_resample['working_hours'] = np.where(
        #     (df_resample['hour'] >= 9) & (df_resample['hour'] < 20), 
        #     1, 
        #     0.1
        # )
        return df_resample
    else:
        return df

# def create_lagged_features(df, target_column, lags):
#     '''
#     Create lag of inputs
#     '''
#     df_with_lags = df.copy()
#     for lag in lags:
#         df_with_lags[f'{target_column}_lag_{lag}'] = df_with_lags[target_column].shift(lag)
#     df_with_lags = df_with_lags.dropna() # Rimuovi le righe con valori NaN introdotti dallo shift
#     return df_with_lags

def create_lagged_features_multiple_columns(df, columns_to_lag, lags):
    '''
    Create lagged features for multiple columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_lag (list): List of column names to create lagged versions of.
        lags (list): List of integers indicating the lag steps.

    Returns:
        pd.DataFrame: A new DataFrame with the lagged features added.
    '''
    df_with_lags = df.copy()
    for column in columns_to_lag:
        for i in range(0, lags):
            # print(i)
            df_with_lags[f'{column}_lag_{i+1}'] = df_with_lags[column].shift(i+1)
    # Drop rows with NaNs from lagging
    df_with_lags.dropna(inplace=True)

    return df_with_lags

def calculate_available_days_per_month(df, date_column='date'):
    """
    Calculates the number of available days for each month in a time series dataframe.
    
    Parameters:
    - df: The dataframe containing temporal data
    - date_column: The name of the column containing dates (default: 'date')
    
    Returns:
    - A dataframe with the count of available days per month
    """
    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Create columns for year and month
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    
    # Count unique days for each year-month combination
    days_per_month = df.groupby(['year', 'month'])[date_column].apply(
        lambda x: x.dt.day.nunique()
    ).reset_index(name='available_days')
    
    # Add a column with month names for readability
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    days_per_month['month_name'] = days_per_month['month'].map(month_names)
    
    return days_per_month


def M_and_V_controls_temperature(
    df: pd.DataFrame, 
    list_temperature_features: list,
    list_energy_features: list, 
    time_training:list, 
    time_testing:list, 
    time_reporting:list, 
    use_temperature_model: bool, 
    temperature_file_path: str, 
    name_col_sim_temp:str, 
    name_temperature_modified: str, 
    time_frequency_resample: list, 
    target: str, 
    time_col_name:str,
    resample_data:bool,
    freq:str,
    model_name: str, 
    folder_path: str, 
    polynomial_degree:int, 
    type_regularization="ridge"):
    """
    M&V model considering a change in temperature set points.

    This function models internal temperatures during the baseline period,
    simulates those temperatures for the reporting period, and uses the
    simulated values to estimate energy consumption as if no changes in control occurred.
    It then calculates the difference between simulated and actual energy usage.

    Parameters
    ----------
    df : pd.DataFrame
        Processed and cleaned monitoring data. Columns must include temperatures and energy values (not power).
    
    list_temperature_features : list of str
        List of column names in `df` representing temperature values to be used in the model.
        Useful for resampling, where averages over different frequencies are applied.
    
    list_energy_features : list of str
        List of column names in `df` representing energy values to be used in the model.
        Useful for summing values over different time frequencies.
    
    use_temperature_model : bool
        If True, replaces real internal temperature values with simulated ones during the reporting period.
        Useful for evaluating the energy consumption model under unchanged temperature control assumptions.
    
    temperature_file_path : str
        Path to the CSV file containing simulated temperature values. The file must have a time-formatted index.
    
    name_col_sim_temp : str
        Name of the column in the simulated temperature file to use for replacing real data.
    
    name_temperature_modified : str
        Name of the temperature sensor column in `df` to replace with simulated values, e.g., 
        "Internal temperature area 2 (Celsius degree)".
    
    time_frequency_resample : list of str
        List of time frequencies for data resampling, e.g., ['30min', 'h', '6h', 'd'].
        The list must include at least one frequency.
    
    target : str
        Target column for the model, e.g., "HVAC energy (kWh)".
    
    time_col_name : str
        Name of the time column in `df`, e.g., "Date and Time".
    
    initial_resample_type : str
        Initial frequency value for resampling, e.g., "h".
    
    type_regularization : str
        Type of regularization to apply. Options are "ridge", "lasso", or "elastic".
    
    model_name : str
        Filename for saving the best model inputs as a pickle file.
    
    folder_path : str
        Directory path where the model files will be stored.

    Returns
    -------
    None
        This function saves model outputs to files and may optionally return evaluation metrics in a full implementation.
    """

    # Define metrics tracking
    best_model = None
    best_metrics = {
        "r2_train": 0,
        "r2_test": 0,
        "mae_train": float('inf'),
        "mae_test": float('inf'),
        "delta_r2": float('inf')
    }
    best_df_training = None
    best_df_testing = None
    best_df_reporting = None
    best_features = None
    best_frequency = None

    # Try different resampling frequencies until we find a good model

    for freq in time_frequency_resample:
        print(f"Trying resample frequency: {freq}")
        
        # Preprocess data with current frequency
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                df_pre_processed = pre_processing_data(df, time_col_name, list_temperature_features, list_energy_features, resample_data, freq)
            else:
                df_pre_processed = pre_processing_data(df.reset_index(), time_col_name, list_temperature_features, list_energy_features, resample_data, freq)
        except Exception as e:
            print(f"Error in preprocessing with frequency {freq}: {e}")
            continue
        
        # Split into training and testing periods
        # df_training = df_pre_processed["01-10-2021":"28-02-2022"]
        # df_testing = df_pre_processed["01-03-2022":"15-03-2022"]
        # df_reporting = df_pre_processed["11-01-2024":"31-03-2025"]
        df_training = df_pre_processed[time_training[0]:time_training[1]]
        df_testing = df_pre_processed[time_testing[0]: time_testing[1]]
        df_reporting = df_pre_processed[time_reporting[0]: time_reporting[1]]
        if use_temperature_model==True:
            df_t1_pred=pd.read_csv(temperature_file_path, index_col=0)
            df_t1_pred.index = pd.to_datetime(df_t1_pred.index)
            for date in df_t1_pred.index:
                if date in df_reporting.index:
                    df_reporting.loc[date, name_temperature_modified] = df_t1_pred.loc[date, name_col_sim_temp]

            
        # df_reporting['11-08-2024':'11-08-2024']
        # df_reporting.plot()
        # df_reporting['Internal temperature area 1 (Celsius degree)'].plot()
        if df_training.empty or df_testing.empty:
            print(f"Training or testing data is empty for frequency {freq}. Skipping.")
            continue
        
        # Get initial features
        features_training = df_training.columns[df_training.columns != target].tolist()
        
        # Build initial model
        model, pred_test_series, mae_train, r2_train, mae_test, r2_test, diagnostics, coef_df = enhanced_linear_regression_noplot(
            df_training, df_testing, target, features_training,  polynomial_degree, type_regularization
        )
        
        # print(f"Initial model metrics - R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}, Delta R2: {r2_train - r2_test:.4f}")
        
        # Check if this model is good enough
        if r2_train >= 0.9 and r2_test >= 0.8 and (r2_train - r2_test) <= 0.1:
            # print(f"Found good model with frequency {freq} without lags")
            if r2_test > best_metrics["r2_test"]:
                best_model = model
                best_metrics = {
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                    "mae_train": mae_train,
                    "mae_test": mae_test,
                    "delta_r2": r2_train - r2_test
                }
                best_df_training = df_training
                best_df_testing = df_testing
                best_df_reporting = df_reporting
                best_features = features_training
                best_frequency = freq
        else:
            
            # Get top important features based on coefficients
            top_features = coef_df.iloc[:3]['Feature'].tolist()
            
            # Define lag ranges based on the resampling frequency
            if freq == "15min":
                lags_to_include = [1, 2, 3,4, 5, 48, 96]  # 15min, 30min, 1h, 2h, 4h, 8h, 24h
            elif freq == "30min":
                lags_to_include = [1, 2, 3, 4, 12, 24, 48]  # 30min, 1h, 2h, 4h, 8h, 24h
            elif freq == "1h":
                lags_to_include = [1, 2, 3, 4, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 4h, 6h, 12h, 24h
            else:  # "2h"
                lags_to_include = [1, 2, 3, 6, 12,24,84]  # 2h, 4h, 6h, 12h, 24h
            
            # Progressively add lags for important features and track improvements
            feature_used = []
            for feature in top_features:
                # print(f"Adding lags for feature: {feature}")
                feature_used.append(feature)
                print(lags_to_include)
                for lag in [1, 2, 3, 4, 6, 12, 24]: 
                    print(lag)
                    # Try adding lags for important features
                    # df_training_lag = df_training.copy()
                    # df_testing_lag = df_testing.copy()
                    # Create lagged features
                    df_training_lag = create_lagged_features_multiple_columns(df_training.copy(), feature_used, lag)
                    df_testing_lag = create_lagged_features_multiple_columns(df_testing.copy(), feature_used, lag)
                    df_reporting_lag = create_lagged_features_multiple_columns(df_reporting.copy(), feature_used, lag)
                    
                    # Get updated features including lags
                    features_with_lags = df_training_lag.columns[df_training_lag.columns != target].tolist()
                    
                    # Build model with lagged features
                    model, pred_test_series, mae_train, r2_train, mae_test, r2_test, diagnostics, coef_df = enhanced_linear_regression_noplot(
                        df_training_lag, df_testing_lag, target, features_with_lags, polynomial_degree, type_regularization="ridge"
                    )
                    
                    # print(f"Lag model metrics for {feature} - R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}, Delta R2: {r2_train - r2_test:.4f}")
                    
                    # Check if this model is better than our previous best
                    if r2_test > best_metrics["r2_test"] and (r2_train - r2_test) < 0.1:
                        best_model = model
                        best_metrics = {
                            "r2_train": r2_train,
                            "r2_test": r2_test,
                            "mae_train": mae_train,
                            "mae_test": mae_test,
                            "delta_r2": r2_train - r2_test
                        }
                        best_df_training = df_training_lag
                        best_df_testing = df_testing_lag
                        best_features = features_with_lags
                        best_frequency = freq
                        best_df_reporting = df_reporting_lag
                    
                    # If we've found a really good model, break early
                    if r2_train >= 0.9 and r2_test >= 0.9 and (r2_train - r2_test) <= 0.15:
                        found_good_model = True
                        # print(f"Found excellent model with lags for {feature}")
                        break
                
                # if found_good_model:
                #     break

    # Output best model information
    print(f"\nBest model results:")
    print(f"Resampling frequency: {best_frequency}")
    print(f"R2 train: {best_metrics['r2_train']:.4f}")
    print(f"R2 test: {best_metrics['r2_test']:.4f}")
    print(f"MAE train: {best_metrics['mae_train']:.4f}")
    print(f"MAE test: {best_metrics['mae_test']:.4f}")
    print(f"Delta R2: {best_metrics['delta_r2']:.4f}")
    if best_features:
        print(f"Number of features: {len(best_features)}")


    data_input_file = data_structure_for_linaer_model(
        best_df_training,best_df_testing, best_df_reporting, target, best_features, type_regularization,
        model_name, folder_path)
    
    return data_input_file


def mean_daily_consumption_and_temperature(df_:pd.DataFrame, reporting_period:list,name_energy:str,name_temperature:str, working_hours:list, year:int=None):
    '''
    Mean of daily consumption and indoor temperature for the reporting period workdays and weekend
    Param
    ------
    df: Dataframe containing the data
    reporting_period: list of the reporting period
    name_temperature: name of the temperature column
    working_hours: list of the working hours [start, end]
    '''

    df_['hour'] = df_.index.hour
    df_['dayofweek'] = df_.index.dayofweek
    df_['year'] = df_.index.year
    # Subset df
    df_reporting = df_.loc[reporting_period[0]:reporting_period[1]]

    df_divided = {
        'df_workdays': df_reporting[(df_reporting['dayofweek'] <= 5)],
        'df_weekend': df_reporting[(df_reporting['dayofweek'] > 5)]    
    }
    df_w = 'df_workdays'
    values = []
    for df_w in df_divided.keys():
        df_week = df_divided[df_w]
                # Calculate the mean of daily consumption and indoor temperature for workdays filtering by hours
        if year is not None:
            df_week_workinghours = df_week[(df_week['hour'] >= working_hours[0]) & (df_week['hour'] <= working_hours[1]) & (df_week['year'] == year)]
            df_week_sleep = df_week[(df_week['hour']  < working_hours[0]) | (df_week['hour'] > working_hours[1]) & (df_week['year'] == year)]
        else:
            df_week_workinghours = df_week[(df_week['hour'] >= working_hours[0]) & (df_week['hour'] <= working_hours[1])]
            df_week_sleep = df_week[(df_week['hour'] < working_hours[0]) | (df_week['hour'] > working_hours[1])]
        # WORKING HOURS
        # Calculate the mean of daily consumption and indoor temperature
        indoor_temp_mean_workinghours = df_week_workinghours.loc[:, [name_temperature]].resample('D').mean().dropna()
        mean_energy_consumption_workinghours = df_week_workinghours.loc[:, [name_energy]].resample('D').sum().dropna()
        mean_energy_consumption_workinghours = mean_energy_consumption_workinghours[mean_energy_consumption_workinghours >0].dropna()
        values.append(mean_energy_consumption_workinghours.mean().values[0].round(2))
        values.append(indoor_temp_mean_workinghours.mean().values[0].round(2))
        #  NO WORKING HOURS
        indoor_temp_mean_sleep = df_week_sleep.loc[:, [name_temperature]].resample('D').mean().dropna()
        mean_energy_consumption_sleep = df_week_sleep.loc[:, [name_energy]].resample('D').sum().dropna()
        mean_energy_consumption_sleep = mean_energy_consumption_sleep[mean_energy_consumption_sleep >0].dropna()
        values.append(mean_energy_consumption_sleep.mean().values[0].round(2))
        values.append(indoor_temp_mean_sleep.mean().values[0].round(2))

    mean_values = {
        'daily_energy_mean_workdays_workhour': values[0],
        'daily_temperature_mean_workdays_workhour': values[1],
        'daily_energy_mean_weekend_workhour': values[4],
        'daily_temperature_mean_weekend_workhour': values[5],
        'daily_energy_mean_workdays_sleep': values[2],
        'daily_temperature_mean_workdays_sleep': values[3],
        'daily_energy_mean_weekend_sleep': values[6],
        'daily_temperature_mean_weekend_sleep': values[7],
    }
    return mean_values


def plotly_actual_vs_predicted(df, col_name_actual, col_name_predicted, target):
    '''
    plotly line chart of predicted value vs actual values
    Param
    ------
    df: dataframe with date time index. 
    '''
    
    fig = go.Figure()

    # Add actual values
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[col_name_actual], 
        mode='lines+markers',
        name='Actual',
        marker=dict(size=6, color='blue'),
        line=dict(width=2)
    ))

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=df.index, 
        y=df[col_name_predicted], 
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

    return fig

def model_generation(
    df:pd.DataFrame=None, 
    time_training:list=None, 
    time_testing:list=None, 
    time_reporting:list=None, 
    list_temperature_features:list=None, 
    list_energy_features:list=None,
    target:str="",
    time_col_name:str="",
    resample_data:bool=True,
    freq:str="h",
    name_temperature_modified:str="",
    time_frequency_resample:list=["h"],
    use_temperature_model:bool=False,
    temperature_file_path:str="",
    name_col_sim_temp:str="",
    folder_path:str="",
    type_regularization:str="",
    polynomial_degree:int=1,
    model_name:str="",
    save_model:bool=True,
    ):

    M_and_V_controls_temperature(
        df=df,
        list_temperature_features=list_temperature_features,
        list_energy_features=list_energy_features,
        time_training=time_training,
        time_testing=time_testing,
        time_reporting=time_reporting,
        use_temperature_model=use_temperature_model,
        temperature_file_path=temperature_file_path,
        name_col_sim_temp=name_col_sim_temp,
        name_temperature_modified=name_temperature_modified,
        time_frequency_resample=time_frequency_resample,
        target=target,
        time_col_name=time_col_name,
        resample_data=resample_data,
        freq=freq,
        model_name=model_name,
        folder_path=folder_path,
        polynomial_degree=polynomial_degree,
        type_regularization=type_regularization,
    )

    # save best model 
    data_model = load_pickle_inputs(f"{folder_path}/{model_name}.pkl")

    # validate the best model through staatistical analysis 
    model, _, _, _, _, _, _, _, df_results, df_results_train, _, df_results_reporting= enhanced_linear_regression(
        df_training=data_model['df_training'],
        df_testing=data_model['df_testing'],
        df_reporting=data_model['df_reporting'],
        target=data_model['target_model'],
        features=data_model['features_model'],
        polynomial_degree=polynomial_degree,
        type_regularization=data_model['regularization_type']
    )
    # save model results
    if save_model:
        df_results_reporting.to_csv(temperature_file_path)

    return data_model, df_results, df_results_train, df_results_reporting


def analysis_of_result(df_, df_results_reporting: pd.DataFrame, df_results_train: pd.DataFrame, name_file_graph: str, temp_predicted: pd.DataFrame, time_reporting:list):
    '''
    Analyze data of model results
    '''

    df_results_reporting_ = df_results_reporting.copy()
    df_results_reporting_.loc[df_results_reporting_['predicted_reporting'] < 0, 'predicted_reporting'] = 0

    sum_df_results_train = df_results_train.loc[:,['actual_train','predicted_train']].sum()
    sum_df_results_reporting = df_results_reporting_.loc[:,['actual_reporting','predicted_reporting']].sum()
    sum_df_results_reporting = pd.DataFrame(sum_df_results_reporting).T
    days = calculate_available_days_per_month(df_results_reporting_.reset_index(), "Date and time")
    sum_df_results_reporting['days'] = days['available_days'].sum()
    sum_df_results_reporting['saving'] = round(sum_df_results_reporting['predicted_reporting'] - sum_df_results_reporting['actual_reporting'], 2)
    sum_df_results_reporting['average actual daily consumption'] = round(sum_df_results_reporting['actual_reporting'] / sum_df_results_reporting['days'], 2)
    sum_df_results_reporting['average predicted daily consumption'] = round(sum_df_results_reporting['predicted_reporting'] / sum_df_results_reporting['days'], 2)
    sum_df_results_reporting['saving daily'] = round(sum_df_results_reporting['saving'] / sum_df_results_reporting['days'], 2)
    sum_df_results_reporting.to_csv(f"report/table_saving_{name_file_graph}.csv")

    # --- Mean values ---
    df_["HVAC energy (kWh)"] = df_["HVAC power (kW)"] * (15*60/3600)
    df_["HVAC energy_global (kWh)"] = df_["Global power (kW)"] * (15*60/3600)
    df_["HVAC energy_no_HVAC (kWh)"] = round(df_["HVAC energy_global (kWh)"] - df_["HVAC energy (kWh)"], 4)
    df_.drop(columns=["HVAC power (kW)", "Global power (kW)"], inplace=True)
    df_.index = pd.to_datetime(df_['Date and time'])
    df_.drop(columns=['Date and time'], inplace=True)

    mean_values_actual = mean_daily_consumption_and_temperature(
        df_=df_,
        reporting_period=time_reporting,
        name_energy="HVAC energy (kWh)",
        name_temperature="Internal temperature area 2 (Celsius degree)",
        working_hours=[9, 20]
    )

    df_results_reporting_['Indoor_temp_2_measured'] = temp_predicted['actual_reporting'].values
    df_results_reporting_['Indoor_temp_2_predicted'] = temp_predicted['predicted_reporting'].values

    mean_values_simulated = mean_daily_consumption_and_temperature(
        df_=df_results_reporting_,
        reporting_period=time_reporting,
        name_energy="predicted_reporting",
        name_temperature="Indoor_temp_2_predicted",
        working_hours=[9, 20]
    )

    pd.DataFrame([mean_values_simulated]).to_csv(f"report/mean_values_simulated_{name_file_graph}.csv")
    pd.DataFrame([mean_values_actual]).to_csv(f"report/mean_values_actual_{name_file_graph}.csv")

    # --- Graphs ---
    df_results_reporting_.to_csv(f"report/data_model_{name_file_graph}.csv")

    BlgGraphs.ipmvp_graph(
        df=df_results_reporting_,
        baseline="actual_reporting",
        adjusted_reporting_period="predicted_reporting",    
        save_mode="path",
        path=f"graphs/{name_file_graph}.html",
        width_graph="1200px",
        height_graph="600px",
        y_name="Energy"
    )

    # Plotly final output
    df_plot = df_results_reporting.resample('d').sum()
    plotly_actual_vs_predicted(df_plot, 'actual_reporting', 'predicted_reporting', 'Energy')

    df_temp = pd.read_csv("data_/ind_temp_pred.csv", index_col=0)
    plotly_actual_vs_predicted(df_temp, 'actual_reporting', 'predicted_reporting', 'Temperature')

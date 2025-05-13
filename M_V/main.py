import pandas as pd
from functions_analysis import *
from models import *

# EXAMPLE PIPELINE
def pipeline_temp_and_energy():
    '''
    Pipeline to get energy model evaluation for shops data
    1. model of indoor temeprature
    2. Predict energy model suring reporting period
    3. save model data in csv to be the input of the neergy model(3)
    4. create energy model in baseline
    5. simulate enrgy model suring reporting period using simulated indoor temeprature
    '''
    # import os
    # os.chdir("/Users/dantonucci/Documents/gitLab/measurement_and_verification/M_V")
    
    # WINTER INPUT
    df_winter = pd.read_csv("data_example/BCG7.csv", sep=';', decimal=",")
    time_training_winter = ["01-10-2021", "28-02-2022"]
    time_testing_winter = ["01-03-2022","15-03-2022"]
    time_reporting_winter = ["11-01-2024","31-03-2025"]
    name_file_graph_winter = "winter_analysis_BCG7"
    temp_file_name_winter = "data_/ind_temp_pred.csv"
    model_name_winter = "HVAC_energy_model_winter"
    temp_predicted_winter = pd.read_csv("data_/ind_temp_pred.csv", index_col=0)

    # SUMMER INPUT
    df_summer = pd.read_csv("data_example/BCG7_cleaned_data_summer.csv")
    time_training_summer = ["07-01-2022", "07-22-2022"]
    time_testing_summer = ["07-22-2022","07-30-2022"]
    time_reporting_summer = ["06-01-2024","07-30-2024"]
    name_file_graph_summer = "summer_analysis_BCG7"
    temp_file_name_summer = "data_/ind_temp_pred_summer.csv"
    model_name_summer = "HVAC_energy_model_summer"
    temp_predicted_summer = pd.read_csv("data_/ind_temp_pred_summer.csv", index_col=0)

    # GENERAL INPUT to be changed according to the possibile choice of winter or summer
    df = df_winter.copy()
    df_= df_winter.copy()
    time_training = time_training_winter
    time_testing = time_testing_winter
    time_reporting = time_reporting_winter
    model_name = model_name_winter
    temperature_file_path = temp_file_name_winter 
    name_file_graph = name_file_graph_winter
    temp_predicted = temp_predicted_winter 
    resample_data = True
    initial_resample_type = "h"
    time_frequency_resample = ["h"]
    use_temperature_model = False
    folder_path = "models"
    type_regularization = "ridge"
    polynomial_degree = 1
    # From Power to energy
    df["HVAC energy (kWh)"] = df["HVAC power (kW)"] * (15*60/3600)
    df["HVAC energy_global (kWh)"] = df["Global power (kW)"] * (15*60/3600)
    df["HVAC energy_no_HVAC (kWh)"] = round(df["HVAC energy_global (kWh)"] - df["HVAC energy (kWh)"],4)
    # Remove Power column
    df = df.drop(columns=["HVAC power (kW)"])
    df = df.drop(columns=["Global power (kW)"])

    indoor_temp  =[
        "Internal temperature area 1 (Celsius degree)",
        "Internal temperature area 2 (Celsius degree)",
        "Internal temperature area 3 (Celsius degree)",
        "Internal temperature area 4 (Celsius degree)",
    ]

    list_temperature_features = [
        "Internal temperature area 2 (Celsius degree)",
        "External temperature (Celsius degree)",    
    ]

    list_energy_features = [
        "HVAC energy (kWh)",
        "HVAC energy_no_HVAC (kWh)"
    ]

    target = "Internal temperature area 2 (Celsius degree)"
    time_col_name = "Date and time"
    name_temperature_modified = "Internal temperature area 2 (Celsius degree)"
    name_col_sim_temp = 'predicted_reporting'

    
    # --------Temperature Model--------
    model_generation(
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
        freq=initial_resample_type,
        model_name=model_name,
        folder_path=folder_path,
        polynomial_degree=polynomial_degree,
        type_regularization=type_regularization,
        save_model=True
    )

    # --- Energy Model ---
    target = "HVAC energy (kWh)"
    use_temperature_model = True

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
        freq=initial_resample_type,
        model_name=model_name,
        folder_path=folder_path,
        polynomial_degree=polynomial_degree,
        type_regularization=type_regularization,
    )

    data_model = load_pickle_inputs(f"{folder_path}/{model_name}.pkl")
    model, _, mae_train, r2_train, mae_test, r2_test, _, _, df_results, df_results_train, _, df_results_reporting = enhanced_linear_regression(
        df_training=data_model['df_training'],
        df_testing=data_model['df_testing'],
        df_reporting=data_model['df_reporting'],
        target=data_model['target_model'],
        features=data_model['features_model'],
        polynomial_degree=polynomial_degree,
        type_regularization=data_model['regularization_type']
    )
    
    analysis_of_result(df_, df_results_reporting, df_results_train, name_file_graph, temp_predicted, time_reporting)

if __name__ == "__main__":
    pipeline_temp_and_energy()

# # visualize data 
# import pandas as pd
# import building_graphs as FcGraph


# df = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/Analisi risparmi energetici/BCBA.csv",
#                 sep=';', decimal=",")
































# # POWER CONSUMPTION 
# FcGraph.simple_linechart(
#     df = df,
#     x = "Date and time",
#     y = "Global power (kW)",
#     y_name = "Global power (kW)",
#     path = "graphs/BCAA_global_power.html"
# )

# # TEMPERATURE
# FcGraph.simple_linechart(
#     df = df,
#     x = "Date and time",
#     y = "Internal temperature area 3 (Celsius degree)",
#     y_name = "Temperature - °C",
#     path = "graphs/BCAA_int_temperature.html"
# )

# # ======================================================
# #           ANOMALIES DETECTION
# # ======================================================
# import matplotlib.pyplot as plt
# from pyod.models.iforest import IForest  # Isolation Forest

# df_detection = df.loc[:,['Date and time', 'Internal temperature area 3 (Celsius degree)']]
# df_detection.dropna(inplace=True)
# # Initialize & Fit Model
# model = IForest()
# model.fit(df_detection[["Internal temperature area 3 (Celsius degree)"]])

# # Predict Anomalies (1 = anomaly, 0 = normal)
# df_detection["anomaly"] = model.predict(df_detection[["Internal temperature area 3 (Celsius degree)"]])

# # Plot Results
# plt.figure(figsize=(12, 6))
# plt.plot(df_detection["Date and time"], df_detection["Internal temperature area 3 (Celsius degree)"], label="Time Series Data", color="blue")
# plt.scatter(df_detection["Date and time"][df_detection["anomaly"] == 1], df_detection["Internal temperature area 3 (Celsius degree)"][df_detection["anomaly"] == 1], 
#             color="red", label="Anomaly", marker="o")
# plt.legend()
# plt.title("Time Series Anomaly Detection using PyOD (Isolation Forest)")
# plt.show()
# #=====
# import stumpy

# m = 24*4
# mp = stumpy.stump(df_detection['Internal temperature area 3 (Celsius degree)'], m)
# import numpy as np
# motif_idx = np.argsort(mp[:, 0])[0]

# print(f"The motif is located at index {motif_idx}")

# fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
# plt.suptitle('Motif (Pattern) Discovery', fontsize='30')
# nearest_neighbor_idx = mp[motif_idx, 1]

# fig, axs = plt.subplots(2)
# import matplotlib.pyplot as plt
# import matplotlib.dates as dates
# from matplotlib.patches import Rectangle
# import datetime as dt
# axs[0].plot(df_detection['Internal temperature area 3 (Celsius degree)'].values)
# axs[0].set_ylabel('Temperature', fontsize='20')
# rect = Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey')
# axs[0].add_patch(rect)
# rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey')
# axs[0].add_patch(rect)
# axs[1].set_xlabel('Time', fontsize ='20')
# axs[1].set_ylabel('Matrix Profile', fontsize='20')
# axs[1].axvline(x=motif_idx, linestyle="dashed")
# axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
# axs[1].plot(mp[:, 0])
# plt.show()


# fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
# plt.suptitle('Motif (Pattern) Discovery', fontsize='30')

# axs[0].plot(df_detection['Internal temperature area 3 (Celsius degree)'].values)
# axs[0].set_ylabel('Temperature', fontsize='20')
# rect = Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey')
# axs[0].add_patch(rect)
# rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey')
# axs[0].add_patch(rect)
# axs[1].set_xlabel('Time', fontsize ='20')
# axs[1].set_ylabel('Matrix Profile', fontsize='20')
# axs[1].axvline(x=motif_idx, linestyle="dashed")
# axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed")
# axs[1].plot(mp[:, 0])
# plt.show()

# mp[motif_idx, 0]

# T = df_detection['Internal temperature area 3 (Celsius degree)'].values
# mp = stumpy.stump(T, m)
# print(mp.P_, mp.I_)

# discord_idx = np.argsort(mp[:, 0])[-1]

# print(f"The discord is located at index {discord_idx}")

# df_detection.iloc[discord_idx,:]


# fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
# plt.suptitle('Discord (Anomaly/Novelty) Discovery', fontsize='30')

# axs[0].plot(df_detection['Internal temperature area 3 (Celsius degree)'].values)
# axs[0].set_ylabel('Temperature', fontsize='20')
# rect = Rectangle((discord_idx, 0), m, 40, facecolor='lightgrey')
# axs[0].add_patch(rect)
# axs[1].set_xlabel('Time', fontsize ='20')
# axs[1].set_ylabel('Matrix Profile', fontsize='20')
# axs[1].axvline(x=discord_idx, linestyle="dashed")
# axs[1].plot(mp[:, 0])
# plt.show()


# # alibe -detect
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pandas as pd
# # import tensorflow as tf

# from alibi_detect.od import OutlierProphet
# from alibi_detect.utils.fetching import fetch_detector
# from alibi_detect.saving import save_detector, load_detector

# df_alibi = df.loc[:, ['Date and time', 'Internal temperature area 3 (Celsius degree)']]
# df_alibi['Date and time'] = pd.to_datetime(df_alibi['Date and time'], format="%d/%m/%Y %H:%M:%S")
# df_alibi.dropna(inplace=True)
# # del df_alibi['Date and time']
# df_alibi.columns = ["ds", "y"]

# df_alibi_train  =df_alibi.iloc[:int(len(df_alibi)*0.9)]
# df_alibi_test = df_alibi.iloc[int(len(df_alibi)*0.9):]
# filepath = 'model_alibi'  # change to directory where model is saved
# detector_name = 'OutlierProphet'
# filepath = os.path.join(filepath, detector_name)

# # initialize, fit and save outlier detector
# od = OutlierProphet(threshold=.9)
# od.fit(df_alibi_train)

# od_preds = od.predict(
#     df_alibi_test,
#     return_instance_score=True,
#     return_forecast=True
# )

# n_periods = 1378
# future = od.model.make_future_dataframe(periods=n_periods, freq='10T', include_history=True)
# forecast = od.model.predict(future)
# fig = od.model.plot(forecast)

# od_preds['data']['forecast']['threshold'] = np.zeros(n_periods)
# od_preds['data']['forecast'][-n_periods:].plot(x='ds', y=['score', 'threshold'])
# plt.title('Outlier score over time')
# plt.xlabel('Time')
# plt.ylabel('Outlier score')
# plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(df_alibi['ds'], df_alibi['y'])
# plt.title('T (in °C) over time')
# plt.xlabel('Time')
# plt.ylabel('T (in °C)')
# plt.show()

# df_alibi.to_csv("df_alibi.csv", index=False)
# # =====
# import pandas as pd
# import matplotlib.pyplot as plt

# from merlion.utils import TimeSeries
# from merlion.models.defaults import DefaultDetectorConfig, DefaultDetector
# from merlion.transform.resample import TemporalResample
# from merlion.evaluate.anomaly import TSADMetric
# from merlion.post_process.threshold import AggregateAlarms

# # Load dataset (timestamp + value)
# df = pd.read_csv("https://raw.githubusercontent.com/salesforce/Merlion/main/examples/data/example.csv", parse_dates=["timestamp"])
# df = df.set_index("timestamp")

# df = df_alibi.copy()
# df.index = df["ds"]
# del df["ds"]
# df['y'] = pd.to_numeric(df['y'], errors='coerce')  # sostituisce i valori non validi con NaN
# df = df.dropna()
# # Plot the raw time series
# df.plot(title="Original Time Series", figsize=(12, 4))
# plt.show()

# # Convert to Merlion TimeSeries format
# ts = TimeSeries.from_pd(df.iloc[:int(len(df)*0.8)])

# # Apply resampling (if needed)
# # transform = TemporalResample(granularity="1h")
# # ts = transform(ts)

# # # Define model + config
# # config = DefaultDetectorConfig()
# # model = DefaultDetector(config)

# from merlion.models.defaults import DefaultDetectorConfig, DefaultDetector
# model = DefaultDetector(DefaultDetectorConfig())
# model.train(train_data=ts[0:1000])
# test_pred = model.get_anomaly_label(time_series=ts)

# # Train model
# model.train(ts)

# # Detect anomalies
# pred = model.get_anomaly_score(ts)
# scores = pred.to_pd()

# # Threshold scores to generate alarms
# model.threshold = AggregateAlarms(alm_threshold=3.0)
# alarms = model.post_process(pred)
# alarms_df = alarms.to_pd()

# # Plot results
# plt.figure(figsize=(12, 4))
# plt.plot(df.index, df["value"], label="Value")
# plt.plot(scores.index, scores["anom_score"], label="Anomaly Score", color="orange")
# plt.scatter(alarms_df.index, [df.loc[t][0] for t in alarms_df.index], color="red", label="Alarms")
# plt.legend()
# plt.title("Merlion Anomaly Detection")
# plt.show()

# # =============================================
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from pyod.models.auto_encoder import AutoEncoder
# from pyod.models.knn import KNN
# from scipy.stats import zscore

# # Preprocessing
# df = df_alibi.copy()
# df['ds'] = pd.to_datetime(df['ds'])
# df = df.set_index('ds')
# df = df.sort_index()
# series = df['y']

# # --- 1. Z-SCORE BASED OUTLIERS ---
# z_scores = zscore(series)
# z_thresh = 3
# z_outliers = np.abs(z_scores) > z_thresh

# # --- 2. ISOLATION FOREST ---
# iso_model = IsolationForest(contamination=0.01, random_state=42)
# iso_preds = iso_model.fit_predict(series.values.reshape(-1, 1))
# iso_outliers = iso_preds == -1

# # --- 3. AUTOENCODER (PyOD) ---
# from pyod.models.auto_encoder_torch import AutoEncoder

# ae_model = AutoEncoder(
#     hidden_neurons=[64, 32, 32, 64],  # correct param for torch-based AE
#     epochs=10,
#     contamination=0.01,
#     verbose=0
# )
# ae_model.fit(series.values.reshape(-1, 1))
# ae_preds = ae_model.predict(series.values.reshape(-1, 1))
# ae_outliers = ae_preds == 1

# # --- 4. KNN (PyOD) ---
# knn_model = KNN(contamination=0.01)
# knn_model.fit(series.values.reshape(-1, 1))
# knn_preds = knn_model.predict(series.values.reshape(-1, 1))
# knn_outliers = knn_preds == 1

# # Visualization function
# def plot_outliers(series, mask, title):
#     plt.figure(figsize=(14, 4))
#     plt.plot(series.index, series.values, label='Value', color='blue')
#     plt.scatter(series.index[mask], series.values[mask], color='red', label='Outlier', s=10)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Plot results
# plot_outliers(series, z_outliers, "Z-Score Based Outliers")
# plot_outliers(series, iso_outliers, "Isolation Forest Outliers")
# plot_outliers(series, ae_outliers, "AutoEncoder (PyOD) Outliers")
# plot_outliers(series, knn_outliers, "KNN (PyOD) Outliers")
#%%
from this import d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from pyod.models.knn import KNN


df = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/Analisi risparmi energetici/BCG7.csv",
                sep=';', decimal=",")

# --- Preprocessing ---
df_summer = df.copy()
df_summer.index = pd.to_datetime(df_summer['Date and time'])
del df_summer['Date and time']

col_name = df_summer.columns[0]
df_summer = df_summer.dropna(axis=1, how='all').dropna()

# --- Filtra per periodi specifici ---
time_training_summer = ["06-01-2022", "07-31-2022"]
time_testing_summer = ["08-01-2022","08-22-2022"]
time_reporting_summer = ["06-01-2024","08-30-2024"]

df_training_summer = df_summer.loc[time_training_summer[0]:time_training_summer[1]]
df_training_summer['Internal temperature area 2 (Celsius degree)'].plot()
time_testing_summer = df_summer.loc[time_testing_summer[0]:time_testing_summer[1]]
time_testing_summer['Internal temperature area 2 (Celsius degree)'].plot()

df_summer.loc[time_reporting_summer[0]:time_reporting_summer[1]]



# time_training_summer = ["01-05-2022", "01-09-2022"]
# time_testing_summer = ["01-03-2022","15-03-2022"]
# time_reporting_summer = ["11-01-2024","31-03-2025"]

# Crea i dataframe per ciascun periodo
df_period_training = df_summer.loc[time_training_summer[0]:time_training_summer[1]]
df_period_testing = df_summer.loc[time_testing_summer[0]:time_testing_summer[1]]
df_period_reporting = df_summer.loc[time_reporting_summer[0]:time_reporting_summer[1]]

# --- Funzione di rilevamento outlier ---
def detect_outliers(series):
    results = {}

    # Z-SCORE
    z_scores = zscore(series)
    z_outliers = np.abs(z_scores) > 3
    results['z'] = z_outliers

    # ISOLATION FOREST
    iso_model = IsolationForest(contamination=0.01, random_state=42)
    iso_outliers = iso_model.fit_predict(series.values.reshape(-1, 1)) == -1
    results['isolation_forest'] = iso_outliers

    # KNN
    knn_model = KNN(contamination=0.01)
    knn_model.fit(series.values.reshape(-1, 1))
    knn_outliers = knn_model.predict(series.values.reshape(-1, 1)) == 1
    results['knn'] = knn_outliers

    # Combined
    results['combined'] = z_outliers & iso_outliers & knn_outliers

    return pd.DataFrame(results, index=series.index)

# --- Visualizzazione con conferma utente ---
import plotly.graph_objects as go

def plot_combined_outliers(df_outliers, title_prefix=''):
    fig = go.Figure()

    # Linea dei valori normali
    fig.add_trace(go.Scatter(
        x=df_outliers['Date and time'],
        y=df_outliers['value'],
        mode='lines',
        name='Valori',
        line=dict(color='blue')
    ))

    # Punti outlier
    outlier_mask = df_outliers['combined']
    fig.add_trace(go.Scatter(
        x=df_outliers['Date and time'][outlier_mask],
        y=df_outliers['value'][outlier_mask],
        mode='markers',
        name=f'Outlier ({sum(outlier_mask)}) rilevati',
        marker=dict(color='red', size=6)
    ))

    fig.update_layout(
        title=f"{title_prefix} - Outlier rilevati da tutti gli algoritmi",
        xaxis_title='Data',
        yaxis_title='Valore',
        legend=dict(x=0, y=1),
        margin=dict(l=50, r=50, t=50, b=50),
        height=400
    )

    fig.show()

    # Interazione utente
    if sum(outlier_mask) > 0:
        print(f"\n{sum(outlier_mask)} outlier rilevati. Vuoi eliminarli? (True/False)")
        user_choice = input("Elimina outlier? (True/False): ").lower() in ['true', 't', 'yes', 'y', '1']
        return user_choice

    return False


# def plot_combined_outliers(df_outliers, title_prefix=''):
#     plt.figure(figsize=(14, 4))
#     plt.plot(df_outliers['Date and time'], df_outliers['value'], label='Valori', color='blue')
#     plt.scatter(df_outliers['Date and time'][df_outliers['combined']], 
#                 df_outliers['value'][df_outliers['combined']], 
#                 color='red', label=f'Outlier ({sum(df_outliers["combined"])}) rilevati', s=10)
#     plt.title(f"{title_prefix} - Outlier rilevati da tutti gli algoritmi")
#     plt.xlabel('Data')
#     plt.ylabel('Valore')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
    
#     # Simulazione di input utente
#     # In un ambiente interattivo (Jupyter/IPython), sostituire con input()
#     if sum(df_outliers["combined"]) > 0:
#         print(f"\n{sum(df_outliers['combined'])} outlier rilevati. Vuoi eliminarli? (True/False)")
        
#         # Per test, sostituisci questa riga con una decisione automatica o un vero input utente
#         user_choice = input("Elimina outlier? (True/False): ").lower() in ['true', 't', 'yes', 'y', '1']
        
#         # Per dimostrazioni o test:
#         # user_choice = True  # Modifica questo valore per simulare diverse scelte dell'utente
        
#         print(f"Scelta: {'Eliminare' if user_choice else 'Mantenere'} gli outlier")
#         return user_choice
#     else:
#         print("Nessun outlier rilevato.")
#         return False

# --- Funzione per processare un singolo dataframe ---
# Convert the index by swapping month and day (01-04 to 04-01)



def process_dataframe(df_period, period_name, auto_remove=False):
    df_cleaned = df_period.copy()
    outliers_info = {}
    
    for column in df_period.columns:
        series = df_period[column]
        if len(series) >= 10:
            # Rilevamento outlier
            outlier_df = detect_outliers(series)
            outlier_df['value'] = series
            
            # Salva gli outlier in una lista
            outlier_indices = outlier_df.index[outlier_df['combined']]
            outlier_values = series[outlier_df['combined']]
            
            if len(outlier_indices) > 0:
                # Visualizzazione e richiesta conferma
                title = f"Colonna: {column} - Periodo: {period_name}"
                
                if auto_remove:
                    # Modalità automatica: rimuove senza chiedere
                    remove_outliers = True
                    print(f"\nColonna {column}: {len(outlier_indices)} outlier rilevati - Rimozione automatica")
                else:
                    
                    # Create a new dataframe with the corrected index
                    # outlier_df.index = pd.DatetimeIndex(new_index)
                    outlier_df = outlier_df.reset_index()
                    plot_combined_outliers(outlier_df, title_prefix=title)
                    remove_outliers = plot_combined_outliers(outlier_df, title_prefix=title)
                
                if remove_outliers:
                    # L'utente ha confermato la rimozione
                    outliers_info[column] = pd.Series(outlier_values, index=outlier_indices)
                    df_cleaned.loc[outlier_indices, column] = np.nan
                    print(f"Outlier rimossi dalla colonna {column}.")
                else:
                    # L'utente ha scelto di mantenere gli outlier
                    print(f"Outlier mantenuti nella colonna {column}.")
            else:
                print(f"Nessun outlier rilevato nella colonna {column}.")
        else:
            print(f"Intervallo troppo breve per l'analisi della colonna {column} nel periodo {period_name}.")
    
    # Creazione del dataframe degli outlier
    all_outliers = pd.DataFrame()
    for column, outliers in outliers_info.items():
        if not outliers.empty:
            temp_df = pd.DataFrame(outliers)
            temp_df.columns = [column]
            all_outliers = pd.concat([all_outliers, temp_df], axis=1)
    
    return df_cleaned, all_outliers, outliers_info

# --- Configurazione della modalità ---
# Impostare su True per rimozione automatica, False per modalità interattiva
automatic_removal = False  # Modificare questo valore per cambiare modalità

# --- Applicazione della funzione a tutti e tre i dataframe ---
print("\n" + "="*50)
print("ELABORAZIONE PERIODO DI TRAINING")
print("="*50)
df_cleaned_training, outliers_training, outliers_info_training = process_dataframe(
    df_period_training, 
    f"Training ({time_training_summer[0]} - {time_training_summer[1]})",
    auto_remove=automatic_removal
)

print("\n" + "="*50)
print("ELABORAZIONE PERIODO DI TESTING")
print("="*50)
df_cleaned_testing, outliers_testing, outliers_info_testing = process_dataframe(
    df_period_testing, 
    f"Testing ({time_testing_summer[0]} - {time_testing_summer[1]})",
    auto_remove=automatic_removal
)

print("\n" + "="*50)
print("ELABORAZIONE PERIODO DI REPORTING")
print("="*50)
df_cleaned_reporting, outliers_reporting, outliers_info_reporting = process_dataframe(
    df_period_reporting, 
    f"Reporting ({time_reporting_summer[0]} - {time_reporting_summer[1]})",
    auto_remove=automatic_removal
)

# --- Sommario ---
def print_summary(period_name, df_original, df_cleaned, outliers_info):
    total_outliers = sum(len(outliers) for outliers in outliers_info.values())
    print(f"\nSommario {period_name}:")
    print(f"Righe originali: {len(df_original)}")
    print(f"Valori NA originali: {df_original.isna().sum().sum()}")
    print(f"Valori NA dopo pulizia: {df_cleaned.isna().sum().sum()}")
    print(f"Totale outlier rimossi: {total_outliers}")
    
    for column, outliers in outliers_info.items():
        print(f"  {column}: {len(outliers)} outlier rimossi")

print("\n" + "="*50)
print("RIEPILOGO FINALE")
print("="*50)
print_summary("TRAINING", df_period_training, df_cleaned_training, outliers_info_training)
print_summary("TESTING", df_period_testing, df_cleaned_testing, outliers_info_testing)
print_summary("REPORTING", df_period_reporting, df_cleaned_reporting, outliers_info_reporting)

# --- Indicazioni per l'uso ---
print("\n" + "="*50)
print("ISTRUZIONI PER L'USO")
print("="*50)
print("Per utilizzare questo script in modo interattivo:")
print("1. Impostare 'automatic_removal = False' per la modalità interattiva")
print("2. Sostituire la riga con il commento '# Per test, sostituisci questa riga...' con:")
print("   user_choice = input(\"Elimina outlier? (True/False): \").lower() in ['true', 't', 'yes', 'y', '1']")
print("3. Eseguire in un ambiente interattivo come Jupyter Notebook o IPython")
print("4. Per la modalità automatica, impostare 'automatic_removal = True'")

# --- Unisci i dataframe puliti ---
df_cleaned_overall = pd.concat([df_cleaned_training, df_cleaned_testing, df_cleaned_reporting])
# Salva il dataframe pulito
df_cleaned_overall.reset_index().to_csv("cleaned_data_summer.csv", index=False)
# %%

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
# Semplice calcolo consderando Gradi giorno 
df_summer = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/cleaned_data_summer.csv")
df = df_summer.copy()
df["HVAC energy (kWh)"] = df["HVAC power (kW)"] * (15*60/3600)
df["HVAC energy_global (kWh)"] = df["Global power (kW)"] * (15*60/3600)
df["HVAC energy_no_HVAC (kWh)"] = round(df["HVAC energy_global (kWh)"] - df["HVAC energy (kWh)"],4)
# Remove Power column
df = df.drop(columns=["HVAC power (kW)"])
df = df.drop(columns=["Global power (kW)"])
df.index = pd.to_datetime(df["Date and time"])
del df['Date and time']
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['year'] = df.index.year
time_reporting_summer = ["06-01-2024","07-30-2024"]

df_reporting = df.loc[time_reporting_summer[0]:time_reporting_summer[1]]
# Calcolo media temperatura interna e consumi per workdays filtrando per orario
df_workdays = df_reporting[(df_reporting['dayofweek'] < 5) & (df_reporting['hour'] >= 9) & (df_reporting['hour'] <= 20) & (df_reporting['year'] == 2024)]
indoor_temp_mean = df_workdays.loc[:, ['Internal temperature area 2 (Celsius degree)']].resample('D').mean().dropna()
mean_energy_consumption = df_workdays.loc[:, ['HVAC energy (kWh)']].resample('D').sum().dropna()
mean_energy_consumption = mean_energy_consumption[mean_energy_consumption >0].dropna()
# media consumo giornaliero 
mean_energy_consumption.mean().values[0].round(2)
# media temperatura interna
indoor_temp_mean.mean().values[0].round(2)

def mean_daily_consumption_and_temperature(df_:pd.DataFrame, time_column:str, reporting_period:list, name_temperature:str, working_hours:list, year:int=None):
    '''
    Mean of daily consumption and indoor temperature for the reporting period
    Param
    ------
    df: Dataframe containing the data
    reporting_period: list of the reporting period
    name_temperature: name of the temperature column
    working_hours: list of the working hours [start, end]
    '''
    df_["HVAC energy (kWh)"] = df_["HVAC power (kW)"] * (15*60/3600)
    df_["HVAC energy_global (kWh)"] = df_["Global power (kW)"] * (15*60/3600)
    df_["HVAC energy_no_HVAC (kWh)"] = round(df_["HVAC energy_global (kWh)"] - df_["HVAC energy (kWh)"],4)
    # Remove Power column
    df_ = df_.drop(columns=["HVAC power (kW)"])
    df_ = df_.drop(columns=["Global power (kW)"])
    df_.index = pd.to_datetime(df_[time_column])
    del df_[time_column]

    df_['hour'] = df_.index.hour
    df_['dayofweek'] = df_.index.dayofweek
    df_['year'] = df_.index.year
    # Subset df
    df_reporting = df_.loc[reporting_period[0]:reporting_period[1]]
    # Calculate the mean of daily consumption and indoor temperature for workdays filtering by hours
    if year is not None:
        df_workdays = df_reporting[(df_reporting['dayofweek'] < 5) & (df_reporting['hour'] >= working_hours[0]) & (df_reporting['hour'] <= working_hours[1]) & (df_reporting['year'] == year)]
    else:
        df_workdays = df_reporting[(df_reporting['dayofweek'] < 5) & (df_reporting['hour'] >= working_hours[0]) & (df_reporting['hour'] <= working_hours[1])]
    # Calculate the mean of daily consumption and indoor temperature
    indoor_temp_mean = df_workdays.loc[:, [name_temperature]].resample('D').mean().dropna()
    mean_energy_consumption = df_workdays.loc[:, ["HVAC energy (kWh)"]].resample('D').sum().dropna()
    mean_energy_consumption = mean_energy_consumption[mean_energy_consumption >0].dropna()
    # Calculate the mean of daily consumption and indoor temperature
    mean_values ={
        'daily_energy_mean': mean_energy_consumption.mean().values[0].round(2),
        'daily_temperature_mean': indoor_temp_mean.mean().values[0].round(2)
    }

    return mean_values

mean_values = mean_daily_consumption_and_temperature(
    df_ = df_winter.copy(), 
    time_column = 'Date and time',
    reporting_period = winte, 
    name_temperature = "Internal temperature area 2 (Celsius degree)",
    working_hours = [9,20],
    year = 2024
)

# COnsumi mensili
df_test = pd.DataFrame({
   "HVAC energy (kWh)": df["HVAC energy (kWh)"],
   "ext_temp": df["External temperature (Celsius degree)"],
   "ind1":df["Internal temperature area 1 (Celsius degree)"],
   "ind2":df["Internal temperature area 2 (Celsius degree)"],
   "ind3":df["Internal temperature area 3 (Celsius degree)"],
   "ind4":df["Internal temperature area 4 (Celsius degree)"]
})
df_test.index= pd.to_datetime(df['Date and time'])

days = calculate_available_days_per_month(df_test.reset_index(), "Date and time")
df_month = pd.DataFrame(
    {
        "HVAC energy (kWh)": df_test['HVAC energy (kWh)'].resample('ME').sum(),
        "ext_temp": df_test['ext_temp'].resample('ME').mean(),
        "ind1": df_test['ind1'].resample('ME').mean(),
        "ind2": df_test['ind2'].resample('ME').mean(),
        "ind3": df_test['ind3'].resample('ME').mean(),
        "ind4": df_test['ind4'].resample('ME').mean()
    }
).dropna()
df_month['available_days'] = days['available_days'].astype(int).values
df_month['daily_average_consumption'] = round(df_month['HVAC energy (kWh)'] / df_month['available_days'],2)

# cacolo cooling degree days 


def calculate_degree_days(df, temp_column='temperature', date_column='date', 
                         hdd_base=18, cdd_base=26):
    """
    Calculates Heating Degree Days (HDD) and Cooling Degree Days (CDD) from temperature data.
    
    Parameters:
    - df: Dataframe containing temperature data
    - temp_column: Column name containing temperature values in Celsius
    - date_column: Column name containing dates
    - hdd_base: Base temperature for heating degree days in Celsius (default: 18°C)
    - cdd_base: Base temperature for cooling degree days in Celsius (default: 26°C)
    
    Returns:
    - A dataframe with daily, monthly, and yearly aggregated HDD and CDD values
    """
    # Ensure date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Calculate daily HDD and CDD
    # HDD: How much below the base temperature (need for heating)
    # CDD: How much above the base temperature (need for cooling)
    df_copy['hdd'] = np.maximum(0, hdd_base - df_copy[temp_column])
    df_copy['cdd'] = np.maximum(0, df_copy[temp_column] - cdd_base)
    
    # Add date components
    df_copy['year'] = df_copy[date_column].dt.year
    df_copy['month'] = df_copy[date_column].dt.month
    df_copy['day'] = df_copy[date_column].dt.day
    
    # If there are multiple temperature readings per day, aggregate to daily values
    daily_data = df_copy.groupby(['year', 'month', 'day']).agg({
        'hdd': 'sum',
        'cdd': 'sum'
    }).reset_index()
    
    # Aggregate to monthly totals
    monthly_data = daily_data.groupby(['year', 'month']).agg({
        'hdd': 'sum',
        'cdd': 'sum'
    }).reset_index()
    
    # Add month names for readability
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    monthly_data['month_name'] = monthly_data['month'].map(month_names)
    
    # Aggregate to yearly totals
    yearly_data = daily_data.groupby('year').agg({
        'hdd': 'sum',
        'cdd': 'sum'
    }).reset_index()
    
    return {
        'daily': daily_data,
        'monthly': monthly_data,
        'yearly': yearly_data
    }
df_ext_temp = df_test.copy().resample('D').mean()
df_ext_temp = df_ext_temp.loc[:, ['ext_temp']].reset_index()
degree_days = calculate_degree_days(df_ext_temp, "ext_temp", "Date and time")
degree_days['monthly']

# Creare una maschera booleana per filtrare i mesi tra 4 e 9
summer_mask = (degree_days['monthly']['month'] >= 4) & (degree_days['monthly']['month'] <= 9)

# Applicare la maschera per filtrare il dataframe
filtered_df = degree_days['monthly'][summer_mask].copy()
filtered_df = filtered_df[filtered_df['year'].isin([2022, 2024])]

df_month['cdd'] = filtered_df['cdd'].values
df_month['hdd'] = filtered_df['hdd'].values

df_month['daily_average_consumption_cdd'] = df_month['daily_average_consumption']/(df_month['cdd'])

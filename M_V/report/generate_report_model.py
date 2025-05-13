import os
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import function_graphs as BlgGraphs
from functions_report_analysis import *

# Local evaluation 
# os.chdir("report")

# ====MAE AND R2====
try:
    model_result = pd.read_csv("model_result_HVAC_energy_model_winter.csv")
except:
    model_result = pd.read_csv("model_result_HVAC_energy_model_summer.csv")
    
mae_train = model_result['mae_train'].values[0].round(2).tolist()
r2_train = model_result['r2_train'].values[0].round(2).tolist()
mae_test = model_result['mae_test'].values[0].round(2).tolist()
r2_test = model_result['r2_test'].values[0].round(2).tolist()


# === DATA ===
# Genera grafico
df = pd.read_csv("data_model_selected.csv")
df.loc[:, ["actual","predicted", "lower_bound", "upper_bound"]] = df.loc[:, ["actual","predicted", "lower_bound", "upper_bound"]].round(2)
df['lower_bound'] = df['lower_bound'].clip(lower=0)
df["band"] = (df["upper_bound"] - df["lower_bound"]).round(2)
df_train = pd.read_csv("data_model_selected_training.csv")
df_train.loc[:, ["actual_train","predicted_train"]] = df_train.loc[:, ["actual_train","predicted_train"]].round(2)

# COefficient importance
coef_df = pd.read_csv("coefficients_model.csv")

# REPORTING WINTER PERIOD
name_file_graph = "winter_analysis_BCG7"
df_results_reporting_ = pd.read_csv(f"data/data_model_{name_file_graph}.csv", index_col=0)
df_results_reporting_.index = pd.to_datetime(df_results_reporting_.index)
df_results_reporting_ = df_results_reporting_.resample("D").sum()
reporting_line_chart_winter = BlgGraphs.ipmvp_graph(
    df =df_results_reporting_,
    baseline = "actual_reporting",
    adjusted_reporting_period = "predicted_reporting",    
    save_mode = "embedded",
    width_graph="1200px",
    height_graph="500px",   
    path="" 
)

# REPORTING SUMMER PERIOD
name_file_graph_summer = "summer_analysis_BCG7"
df_results_reporting_summer = pd.read_csv(f"data/data_model_{name_file_graph_summer}.csv", index_col=0)
df_results_reporting_summer.index = pd.to_datetime(df_results_reporting_summer.index)
df_results_reporting_summer = df_results_reporting_summer.resample("D").sum()
reporting_line_chart_summer = BlgGraphs.ipmvp_graph(
    df =df_results_reporting_summer ,
    baseline = "actual_reporting",
    adjusted_reporting_period = "predicted_reporting",    
    save_mode = "embedded",
    width_graph="1200px",
    height_graph="500px",   
    path="" 
)

linechart_actual_vs_predicted_train = BlgGraphs.simple_linechart(
    df = df_train,
    x = "Date and time",
    y = "actual_train",
    y_name = "actual_train",
    y2 = "predicted_train",
    y2_name = "predicted_train",
    y2_second_axis = False, 
    save_mode = "embedded",
    width_graph="1200px",
    height_graph="500px",
    confidence_interval = False,
    y_lower_ci = "",
    y_higher_ci = ""
)

linechart_actual_vs_predicted_test = BlgGraphs.simple_linechart(
    df = df,
    x = "Date and time",
    y = "actual",
    y_name = "actual",
    y2 = "predicted",
    y2_name = "predicted",
    y2_second_axis = False, 
    save_mode = "embedded",
    width_graph="1200px",
    height_graph="500px",
    confidence_interval = True,
    y_lower_ci = "lower_bound",
    y_higher_ci = "upper_bound"
)

metrics_actual_vs_predicted = BlgGraphs.scatter_actual_vs_predicted(
    df = df,
    x = "actual",
    y = "predicted",
    save_mode = "embedded",
    width_graph="600px",
    height_graph="350px",
)

residual_distribution = BlgGraphs.residual_distribution(
    df = df, 
    actual_values = "actual", 
    predicted_values = "predicted", 
    save_mode = "embedded", 
    width_graph="600px", 
    height_graph="350px",
    toolbox_option = "Zoom"
)

residual_vs_predicted_values = BlgGraphs.residuals_vs_predicted_values(
    df = df, 
    save_mode = 'embedded', 
    width_graph="600px", 
    height_graph="350px", 
    path=""
)

fetaures_importance = BlgGraphs.features_importance(
    coef_df = coef_df, 
    width_graph = "600px", 
    height_graph = "350px", 
    save_mode = "embedded",
    path = "",
)



# Carica e compila template HTML
env = Environment(loader=FileSystemLoader("."))
template = env.get_template("template_model_generation.html")
html_output = template.render(
    mae_train=mae_train, 
    r2_train=r2_train, 
    mae_test=mae_test, 
    r2_test=r2_test, 
    line_chart_train=linechart_actual_vs_predicted_train,
    line_chart_test=linechart_actual_vs_predicted_test,
    actual_vs_predicted_scatter = metrics_actual_vs_predicted, 
    residual_distribution = residual_distribution,
    residual_vs_predicted_values = residual_vs_predicted_values,
    fetaures_importance = fetaures_importance,
    avg_indoor_temperature_winter_actual =20,
)

# Salva il file
with open("model_report.html", "w", encoding="utf-8") as f:
    f.write(html_output)

print("✅ Report HTML creato: model_report.html")

import os
from jinja2 import Environment, FileSystemLoader
# from charts_report import generate_line_chart
import pandas as pd
import building_graphs as BlgGraphs
from functions_report_analysis import *

# === FUNCTIONS ===
def data_table(data:list):
    '''
    data: list of dict
    e.g: data = [
        {
            "period": "2024",
            "measured_energy": 1250.75,
            "simulated_energy": 1520.30,
            "energy_saving": 269.55,
            "economic_saving": 53.91
        }
    ]
    '''
    rows_html = ""
    for row in data:
        # Determine if savings are positive or negative for styling
        energy_class = "positive-savings" if row["energy_saving"] > 0 else "negative-savings"
        economic_class = "positive-savings" if row["economic_saving"] > 0 else "negative-savings"
        
        # Format the row HTML
        row_html = f"""
                <tr>
                    <td>{row["period"]}</td>
                    <td>{row["measured_energy"]:.2f}</td>
                    <td>{row["simulated_energy"]:.2f}</td>
                    <td class="{energy_class}">{row["energy_saving"]:.2f}</td>
                    <td class="{economic_class}">{row["economic_saving"]:.2f}</td>
                </tr>
        """
        rows_html += row_html

    return rows_html


def gauge(saving):
    from pyecharts import options as opts
    from pyecharts.charts import Gauge, Grid
    c = (
        Gauge(init_opts=opts.InitOpts(width="500px", height="400px"))
        .add("", [("Saving - %", saving)], radius="100%", center= ['50%', '60%'], 
        detail_label_opts=opts.LabelOpts(
            font_size=30,
            color= "rgb(52, 71, 103)",
            font_weight="bold"
        )
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="Energy saving"))
    )
    return c.render_embed()

# === DATA ===
# Genera grafico
df = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/data_model_selected.csv")
df.loc[:, ["actual","predicted", "lower_bound", "upper_bound"]] = df.loc[:, ["actual","predicted", "lower_bound", "upper_bound"]].round(2)
df['lower_bound'] = df['lower_bound'].clip(lower=0)
df["band"] = (df["upper_bound"] - df["lower_bound"]).round(2)
df_train = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/data_model_selected_training.csv")
df_train.loc[:, ["actual_train","predicted_train"]] = df_train.loc[:, ["actual_train","predicted_train"]].round(2)

# COefficient importance
coef_df = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/coefficients_model.csv")

# REPORTING WINTER PERIOD
name_file_graph = "winter_analysis_BCG7"
df_results_reporting_ = pd.read_csv(f"/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/data/data_model_{name_file_graph}.csv", index_col=0)
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
df_results_reporting_summer = pd.read_csv(f"/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/data/data_model_{name_file_graph_summer}.csv", index_col=0)
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

# TABLE SAVING
# Table Winter 
# calculate saving
energy_cost = 0.25
try: 
    data_table_saving_winter = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/data/table_saving_winter_analysis_BCG7.csv", index_col=0)
    data_table_saving_winter['actual reporting cost'] = data_table_saving_winter['actual_reporting'] * energy_cost
    data_table_saving_winter['predicted reporting cost'] = data_table_saving_winter['predicted_reporting'] * energy_cost
    data_table_saving_winter['saving cost'] = data_table_saving_winter['saving'] * energy_cost
    data_table_saving_winter['percentage_saving'] = round((data_table_saving_winter['saving'] / data_table_saving_winter['predicted_reporting'])*100,2)
    data_winter = [
        {
            "period": "2024",
            "measured_energy": data_table_saving_winter['actual_reporting'].sum(),
            "simulated_energy": data_table_saving_winter['predicted_reporting'].sum(),
            "energy_saving": data_table_saving_winter['saving'].sum(),
            "economic_saving": data_table_saving_winter['saving cost'].sum(),
        }
    ]
    rows_html_winter = data_table(data_winter)
    gauge_winter = gauge(data_table_saving_winter['percentage_saving'].values[0])
except: 
    rows_html_winter = ""
    gauge_winter = gauge(0)

# Table Summer 
try: 
    data_table_saving_summer = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/data/table_saving_summer_analysis_BCG7.csv", index_col=0)
    data_summer = [
        {
            "period": "2024",
            "measured_energy": data_table_saving_summer['actual_reporting'].sum(),
            "simulated_energy": data_table_saving_summer['predicted_reporting'].sum(),
            "energy_saving": data_table_saving_summer['saving'].sum(),
            "economic_saving": data_table_saving_summer['saving cost'].sum(),
        }
    ]
    rows_html_summer = data_table(data_summer)
    gauge_summer = gauge(data_table_saving_summer['percentage_saving'].values[0])
except: 
    rows_html_summer = ""
    gauge_summer = gauge(0)
    

# METRICS
# WINTER
df_metrics_actual_summer = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/mean_values_actual_summer_analysis_BCG7.csv", index_col=0)
df_metrics_simulated_summer = pd.read_csv("/Users/dantonucci/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/00_GitHubProject/MODERATE_building_benchmarking/anomalies_detection/M_V/report/mean_values_simulated_summer_analysis_BCG7.csv", index_col=0)
# WORKDAYS
ind_temp_workdays_workinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_workdays_workhour'].values[0]
ind_temp_workdays_workinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_workdays_workhour'].values[0]
ind_temp_workdays_closinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_workdays_sleep'].values[0]
ind_temp_workdays_closinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_workdays_sleep'].values[0]
# WEEKEND
ind_temp_weekend_workinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_weekend_workhour'].values[0]
ind_temp_weekend_workinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_weekend_workhour'].values[0]
ind_temp_weekend_closinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_weekend_sleep'].values[0]
ind_temp_weekend_closinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_weekend_sleep'].values[0]

# WORKDAYS
ind_energy_workdays_workinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_workdays_workhour'].values[0]
ind_energy_workdays_workinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_workdays_workhour'].values[0]
ind_energy_workdays_closinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_workdays_sleep'].values[0]
ind_energy_workdays_closinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_workdays_sleep'].values[0]
# WEEKEND
ind_energy_weekend_workinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_weekend_workhour'].values[0]
ind_energy_weekend_workinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_weekend_workhour'].values[0]
ind_energy_weekend_closinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_weekend_sleep'].values[0]
ind_energy_weekend_closinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_weekend_sleep'].values[0]
try:
    # ---- WINTER ----
    df_metrics_actual = pd.read_csv("mean_values_actual_winter_analysis_BCG7.csv", index_col=0)
    df_metrics_simulated = pd.read_csv("mean_values_simulated_winter_analysis_BCG7.csv", index_col=0)
    # WORKDAYS
    ind_temp_workdays_workinghours_actual = df_metrics_actual['daily_temperature_mean_workdays_workhour'].values[0]
    ind_temp_workdays_workinghours_simulated = df_metrics_simulated['daily_temperature_mean_workdays_workhour'].values[0]
    ind_temp_workdays_closinghours_actual = df_metrics_actual['daily_temperature_mean_workdays_sleep'].values[0]
    ind_temp_workdays_closinghours_simulated = df_metrics_simulated['daily_temperature_mean_workdays_sleep'].values[0]
    # WEEKEND
    ind_temp_weekend_workinghours_actual = df_metrics_actual['daily_temperature_mean_weekend_workhour'].values[0]
    ind_temp_weekend_workinghours_simulated = df_metrics_simulated['daily_temperature_mean_weekend_workhour'].values[0]
    ind_temp_weekend_closinghours_actual = df_metrics_actual['daily_temperature_mean_weekend_sleep'].values[0]
    ind_temp_weekend_closinghours_simulated = df_metrics_simulated['daily_temperature_mean_weekend_sleep'].values[0]

    # WORKDAYS
    ind_energy_workdays_workinghours_actual = df_metrics_actual['daily_energy_mean_workdays_workhour'].values[0]
    ind_energy_workdays_workinghours_simulated = df_metrics_simulated['daily_energy_mean_workdays_workhour'].values[0]
    ind_energy_workdays_closinghours_actual = df_metrics_actual['daily_energy_mean_workdays_sleep'].values[0]
    ind_energy_workdays_closinghours_simulated = df_metrics_simulated['daily_energy_mean_workdays_sleep'].values[0]
    # WEEKEND
    ind_energy_weekend_workinghours_actual = df_metrics_actual['daily_energy_mean_weekend_workhour'].values[0]
    ind_energy_weekend_workinghours_simulated = df_metrics_simulated['daily_energy_mean_weekend_workhour'].values[0]
    ind_energy_weekend_closinghours_actual = df_metrics_actual['daily_energy_mean_weekend_sleep'].values[0]
    ind_energy_weekend_closinghours_simulated = df_metrics_simulated['daily_energy_mean_weekend_sleep'].values[0]

    # ---- SUMMER ----
    df_metrics_actual_summer = pd.read_csv("mean_values_actual_summer_analysis_BCG7.csv", index_col=0)
    df_metrics_simulated_summer = pd.read_csv("mean_values_simulated_summer_analysis_BCG7.csv", index_col=0)
    # WORKDAYS
    ind_temp_workdays_workinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_workdays_workhour'].values[0]
    ind_temp_workdays_workinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_workdays_workhour'].values[0]
    ind_temp_workdays_closinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_workdays_sleep'].values[0]
    ind_temp_workdays_closinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_workdays_sleep'].values[0]
    # WEEKEND
    ind_temp_weekend_workinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_weekend_workhour'].values[0]
    ind_temp_weekend_workinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_weekend_workhour'].values[0]
    ind_temp_weekend_closinghours_actual_summer = df_metrics_actual_summer['daily_temperature_mean_weekend_sleep'].values[0]
    ind_temp_weekend_closinghours_simulated_summer = df_metrics_simulated_summer['daily_temperature_mean_weekend_sleep'].values[0]

    # WORKDAYS
    ind_energy_workdays_workinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_workdays_workhour'].values[0]
    ind_energy_workdays_workinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_workdays_workhour'].values[0]
    ind_energy_workdays_closinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_workdays_sleep'].values[0]
    ind_energy_workdays_closinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_workdays_sleep'].values[0]
    # WEEKEND
    ind_energy_weekend_workinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_weekend_workhour'].values[0]
    ind_energy_weekend_workinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_weekend_workhour'].values[0]
    ind_energy_weekend_closinghours_actual_summer = df_metrics_actual_summer['daily_energy_mean_weekend_sleep'].values[0]
    ind_energy_weekend_closinghours_simulated_summer = df_metrics_simulated_summer['daily_energy_mean_weekend_sleep'].values[0]
except:
    # WINTER
    ind_temp_workdays_workinghours_actual = ""
    ind_temp_workdays_workinghours_simulated = ""
    ind_temp_workdays_closinghours_actual = ""
    ind_temp_workdays_closinghours_simulated = ""
    ind_temp_weekend_workinghours_actual = ""
    ind_temp_weekend_workinghours_simulated = ""
    ind_temp_weekend_closinghours_actual = ""
    ind_temp_weekend_closinghours_simulated = ""
    ind_energy_workdays_workinghours_actual = ""
    ind_energy_workdays_workinghours_simulated = ""
    ind_energy_workdays_closinghours_actual = ""
    ind_energy_workdays_closinghours_simulated = ""
    ind_energy_weekend_workinghours_actual = ""
    ind_energy_weekend_workinghours_simulated = ""
    ind_energy_weekend_closinghours_actual = ""
    ind_energy_weekend_closinghours_simulated = ""
    # SUMMER
    ind_temp_workdays_workinghours_actual_summer = ""
    ind_temp_workdays_workinghours_simulated_summer = ""
    ind_temp_workdays_closinghours_actual_summer = ""
    ind_temp_workdays_closinghours_simulated_summer = ""
    ind_temp_weekend_workinghours_actual_summer = ""
    ind_temp_weekend_workinghours_simulated_summer = ""
    ind_temp_weekend_closinghours_actual_summer = ""
    ind_temp_weekend_closinghours_simulated_summer = ""
    ind_energy_workdays_workinghours_actual_summer = ""
    ind_energy_workdays_workinghours_simulated_summer = ""
    ind_energy_workdays_closinghours_actual_summer = ""
    ind_energy_workdays_closinghours_simulated_summer = ""
    ind_energy_weekend_workinghours_actual_summer = ""
    ind_energy_weekend_workinghours_simulated_summer = ""
    ind_energy_weekend_closinghours_actual_summer = ""
    ind_energy_weekend_closinghours_simulated_summer = ""

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
template = env.get_template("template.html")
mae=2
r2=2
html_output = template.render(
    mae=mae, 
    r2=r2, 
    # WInter reporting
    reporting_line_chart_winter = reporting_line_chart_winter,
    table_rows_winter = rows_html_winter,
    avg_temperature = "20",
    gauge_winter=gauge_winter,
    # SUmmer reporting
    reporting_line_chart_summer = reporting_line_chart_summer,
    table_rows_summer = rows_html_summer,
    gauge_summer=gauge_summer,
    # Metrics
    # Winter
    ind_temp_workdays_workinghours_actual = ind_temp_workdays_workinghours_actual,
    ind_temp_workdays_workinghours_simulated = ind_temp_workdays_workinghours_simulated,
    ind_temp_workdays_closinghours_actual = ind_temp_workdays_closinghours_actual,
    ind_temp_workdays_closinghours_simulated = ind_temp_workdays_closinghours_simulated,
    ind_temp_weekend_workinghours_actual = ind_temp_weekend_workinghours_actual,
    ind_temp_weekend_workinghours_simulated = ind_temp_weekend_workinghours_simulated,
    ind_temp_weekend_closinghours_actual = ind_temp_weekend_closinghours_actual,
    ind_temp_weekend_closinghours_simulated = ind_temp_weekend_closinghours_simulated,
    ind_energy_workdays_workinghours_actual = ind_energy_workdays_workinghours_actual,
    ind_energy_workdays_workinghours_simulated = ind_energy_workdays_workinghours_simulated,
    ind_energy_workdays_closinghours_actual = ind_energy_workdays_closinghours_actual,
    ind_energy_workdays_closinghours_simulated = ind_energy_workdays_closinghours_simulated,
    ind_energy_weekend_workinghours_actual = ind_energy_weekend_workinghours_actual,
    ind_energy_weekend_workinghours_simulated = ind_energy_weekend_workinghours_simulated,
    ind_energy_weekend_closinghours_actual = ind_energy_weekend_closinghours_actual,
    ind_energy_weekend_closinghours_simulated = ind_energy_weekend_closinghours_simulated,
    # Summer
    ind_temp_workdays_workinghours_actual_summer = ind_temp_workdays_workinghours_actual_summer,
    ind_temp_workdays_workinghours_simulated_summer = ind_temp_workdays_workinghours_simulated_summer,
    ind_temp_workdays_closinghours_actual_summer = ind_temp_workdays_closinghours_actual_summer,
    ind_temp_workdays_closinghours_simulated_summer = ind_temp_workdays_closinghours_simulated_summer,
    ind_temp_weekend_workinghours_actual_summer = ind_temp_weekend_workinghours_actual_summer,
    ind_temp_weekend_workinghours_simulated_summer = ind_temp_weekend_workinghours_simulated_summer,
    ind_temp_weekend_closinghours_actual_summer = ind_temp_weekend_closinghours_actual_summer,
    ind_temp_weekend_closinghours_simulated_summer = ind_temp_weekend_closinghours_simulated_summer,
    ind_energy_workdays_workinghours_actual_summer = ind_energy_workdays_workinghours_actual_summer,
    ind_energy_workdays_workinghours_simulated_summer = ind_energy_workdays_workinghours_simulated_summer,
    ind_energy_workdays_closinghours_actual_summer = ind_energy_workdays_closinghours_actual_summer,
    ind_energy_workdays_closinghours_simulated_summer = ind_energy_workdays_closinghours_simulated_summer,
    ind_energy_weekend_workinghours_actual_summer = ind_energy_weekend_workinghours_actual_summer,
    ind_energy_weekend_workinghours_simulated_summer = ind_energy_weekend_workinghours_simulated_summer,
    ind_energy_weekend_closinghours_actual_summer = ind_energy_weekend_closinghours_actual_summer,
    ind_energy_weekend_closinghours_simulated_summer = ind_energy_weekend_closinghours_simulated_summer,
    # Training
    line_chart_train=linechart_actual_vs_predicted_train,
    line_chart_test=linechart_actual_vs_predicted_test,
    actual_vs_predicted_scatter = metrics_actual_vs_predicted, 
    residual_distribution = residual_distribution,
    residual_vs_predicted_values = residual_vs_predicted_values,
    fetaures_importance = fetaures_importance,
    avg_indoor_temperature_winter_actual =20,
)

# Salva il file
with open("measurement_and_verification_report.html", "w", encoding="utf-8") as f:
    f.write(html_output)

print("✅ Report HTML creato: measurement_and_verification_report.html")

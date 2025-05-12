import pandas as pd 
import numpy as np
from pyecharts.charts import Line, Scatter, Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType
import scipy.stats as stats

def ipmvp_graph(df:pd.DataFrame, baseline:str, adjusted_reporting_period:str,save_mode:str, path: str=None, width_graph:str="1200px", height_graph:str="600px", y_name:str="Energy"):
    '''
    Html chart to visualize the ipmvp procedure 
    
    Param
    ------
    df: dataframe
        dataframe with data and datetime index
    baseline: str
        name of the baseline column
    adjusted_reporting_period: str
        name of the adjusted reporting period column
    save_mode: str
        "path":Save file in a defined path
        "embedded": Save file to be embedded in a report 
    path : str
        Path to save the HTML file.
    width_graph : str, optional
        Width of the chart (default is "1200px").
    height_graph : str, optional
        Height of the chart (default is "600px").
    y_name; str
        Nmae of the y axis

    Returns
    -------
    str
        Saves HTML file to a specific path and returns the path.
    '''
    x = df.index.strftime('%d/%m/%Y %H:%M:%S').tolist()
    lower = df[baseline].tolist()
    upper = df[adjusted_reporting_period].tolist()

    # white Area (till lower limit)
    white_area = lower  # Dal fondo fino al lower limit

    # Green area (between lower and upper)
    green_area = [u - l for u, l in zip(upper, lower)]
    green_area_modified = []
    for element in green_area:
        if element <0:
            green_area_modified.append(0)
        else:
            green_area_modified.append(element)

    # Graph
    line = (
        Line(init_opts=opts.InitOpts(width=width_graph, height=height_graph, theme=ThemeType.LIGHT))
        .add_xaxis(x)
        # White area 
        .add_yaxis(
            "",
            white_area,
            stack="total",
            areastyle_opts=opts.AreaStyleOpts(opacity=1.0, color="#FFFFFF"),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=0),
        )
        # Green area (between lower and upper)
        .add_yaxis(
            "Energy saving",
            green_area_modified,
            stack="total",
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5, color="#00C853"),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=0),
        )
        .add_yaxis(
            "Adjusted baseline energy",
            upper,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#1B5E20")
        )
        .add_yaxis(
            "Lower Limit",
            lower,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#BDBDBD")
        )
        # Visual and interactive options
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis"),  # Enable tooltips
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],  # Enable zoom inside
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                orient="horizontal",
                pos_left="right",
                feature={
                    # Save as Image tool
                    "saveAsImage": opts.ToolBoxFeatureSaveAsImageOpts(
                        title="Save as Image",
                        type_="png",
                        pixel_ratio=2
                    ),
                    # Restore chart tool
                    "restore": opts.ToolBoxFeatureRestoreOpts(
                        title="Restore"
                    ),
                    # Data View tool
                    "dataView": opts.ToolBoxFeatureDataViewOpts(
                        title="Data View", 
                        is_show=True,
                        lang=["Data View", "Close", "Refresh"]
                    ),
                    
                    # Data Zoom tool
                    "dataZoom": opts.ToolBoxFeatureDataZoomOpts(
                        is_show=True,
                        zoom_title="Zoom In",back_title="Zoom Out"
                    ),
                    # Magic Type tool for switching chart types
                    "magicType": opts.ToolBoxFeatureMagicTypeOpts(
                        is_show=False,
                    ),
                }
            ),
            yaxis_opts=opts.AxisOpts(name=y_name, type_="value", position="left"),  # Primary Y-axis (left)
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )
    # Save chart as HTML
    if save_mode == "path":
        return line.render(path)
    elif save_mode == "embedded":
        return line.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")




def simple_linechart(
    df: pd.DataFrame, x: str, y: str, y_name: str, save_mode:str, y2: str = None, y2_name: str = None,  y2_second_axis:bool=True, 
    path: str=None, width_graph:str="1200px", height_graph:str="600px", confidence_interval:bool = False,
    y_lower_ci: str = None, y_higher_ci: str = None
):
    """
    Simple line chart using pyecharts with optional second line on a secondary Y-axis.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with data.
    x : str
        Name of the df column for the x-axis.
    y : str
        Name of the df column for the first y-axis.
    y_name : str
        Name of the dataset to be visualized (first line).
    save_mode: str
        "path":Save file in a defined path
        "embedded": Save file to be embedded in a report 
    path : str
        Path to save the HTML file.
    y2 : str, optional
        Name of the df column for the second y-axis (optional).
    y2_name : str, optional
        Name of the second dataset to be visualized (optional).
    width_graph : str, optional
        Width of the chart (default is "1200px").
    height_graph : str, optional
        Height of the chart (default is "600px").

    Returns
    -------
    str
        Saves HTML file to a specific path and returns the path.
    """
    
    # Create Line Chart
    line_chart = (
        Line(init_opts=opts.InitOpts(width=width_graph, height=height_graph, theme=ThemeType.LIGHT))
        .add_xaxis(df[x].tolist())  # X-axis
        .add_yaxis(
            series_name=y_name,
            y_axis=df[y].tolist(),
            yaxis_index=0,  # Primary Y-axis (Left)
            is_smooth=True,  # Smooth line for better visualization
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False)
        )
    )
    if y2_second_axis:
        axis_index = 1 # right axis
        # Configure the secondary Y-axis
        line_chart.extend_axis(
            yaxis=opts.AxisOpts(
                name=y2_name,
                type_="value",
                position="right",
                axisline_opts=opts.AxisLineOpts(linestyle_opts=opts.LineStyleOpts(color="red")),
            )
        )
    else:
        axis_index = 0

    # Add second line on secondary Y-axis (right) if provided
    if y2 and y2_name:
        line_chart.add_yaxis(
            series_name=y2_name,
            y_axis=df[y2].tolist(),
            yaxis_index=axis_index,  # Secondary Y-axis (Right)
            is_smooth=True,  # Smooth line
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False)
        )

    # Add area chart form confidence interval for example
    if confidence_interval ==  True: 
        line_chart.add_yaxis(
            series_name="lower band",
            y_axis=df[y_lower_ci].tolist(),
            yaxis_index=0,  # Secondary Y-axis (Right)
            is_smooth=True,  # Smooth line
            is_symbol_show=False,
            stack="confidence-band",
            linestyle_opts=opts.LineStyleOpts(opacity=0.1, color = "#ccc"),
        )
        line_chart.add_yaxis(
            series_name="upper band",
            y_axis=df[y_higher_ci].tolist(),
            yaxis_index=0,  # Secondary Y-axis (Right)
            is_smooth=True,  # Smooth line
            is_symbol_show=False,
            stack="confidence-band",
            linestyle_opts=opts.LineStyleOpts(opacity=0.1, color="#ccc"),
            areastyle_opts=opts.AreaStyleOpts(color="#ccc", opacity = 0.8),
        )
    # Set global options
    line_chart.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),  # Enable tooltips
        datazoom_opts=[opts.DataZoomOpts(type_="inside")],  # Enable zoom inside
        toolbox_opts=opts.ToolboxOpts(
        is_show=True,
        orient="horizontal",
        pos_left="right",
        feature={
            # Save as Image tool
            "saveAsImage": opts.ToolBoxFeatureSaveAsImageOpts(
                title="Save as Image",
                type_="png",
                pixel_ratio=2
            ),
            # Restore chart tool
            "restore": opts.ToolBoxFeatureRestoreOpts(
                title="Restore"
            ),
            # Data View tool
            "dataView": opts.ToolBoxFeatureDataViewOpts(
                title="Data View", 
                is_show=True,
                lang=["Data View", "Close", "Refresh"]
            ),
            
            # Data Zoom tool
            "dataZoom": opts.ToolBoxFeatureDataZoomOpts(
                is_show=True,
                zoom_title="Zoom In",back_title="Zoom Out"
            ),
            # Magic Type tool for switching chart types
            "magicType": opts.ToolBoxFeatureMagicTypeOpts(
                is_show=False,
            ),
        }
    ),
        yaxis_opts=opts.AxisOpts(name=y_name, type_="value", position="left"),  # Primary Y-axis (left)
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
    )
    
    # Save chart as HTML
    if save_mode == "path":
        return line_chart.render(path)
    elif save_mode == "embedded":
        return line_chart.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")


def scatter_with_regression(df:pd.DataFrame, x:str, y:str, x_name:str, y_name:str, path:str):
    '''
    Scatter plot with regression line using pyecharts
    Param
    ------
    df: dataframe with data
    x: name df column for x axis
    y: name df column for y axis
    x_name: name of the x axis
    y_name: name of the y axis
    path: path to save html file

    Return
    -------
    save html file to specific path 
    '''

    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x].values, df[y].values)

    # Create regression line data points
    line_x = np.linspace(min(df[x].values), max(df[x].values), 100)  # More points for smoother line
    line_y = slope * line_x + intercept

    # Create a DataFrame for the regression line
    df_regression = pd.DataFrame({
        'x': line_x,
        'y': line_y
    })
    # Create scatter chart with regression line
    scatter_chart = (
            Scatter(init_opts=opts.InitOpts(width="1200px", height="600px", theme=ThemeType.LIGHT))
            .add_xaxis(df[x].tolist())  # X-axis for scatter points
            .add_yaxis(
                "Data Points", 
                df[y].tolist(),
                symbol_size=10,
                label_opts=opts.LabelOpts(is_show=False)
            )  # Y-axis for scatter points
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Scatter Plot with Regression Line", 
                subtitle=f"y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})"
            ),
            xaxis_opts=opts.AxisOpts(
                name="X", 
                type_="value", 
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                name="Y", 
                type_="value",
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),  # Enable Tooltips
            datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],  # Enable Zoom
        )
    )

    # Add the regression line using Line chart
    line_chart = (
        Line()
        .add_xaxis(df_regression["x"].tolist())  # X-axis for regression line
        .add_yaxis(
            "Regression Line", 
            df_regression["y"].tolist(),
            symbol="none",  # No symbols on the line
            is_smooth=True,  # Make the line smooth
            linestyle_opts=opts.LineStyleOpts(width=2, type_="solid", color="red"),
            label_opts=opts.LabelOpts(is_show=False)
        )  # Y-axis for regression line
    )

    # Overlap the scatter plot and line chart
    scatter_chart.overlap(line_chart)

    # Save the chart
    if save_mode == "path":
        return line_chart.render(path)
    elif save_mode == "embedded":
        return line_chart.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")


def ipmvp_graph(df:pd.DataFrame, baseline:str, adjusted_reporting_period:str,save_mode:str, path: str=None, width_graph:str="1200px", height_graph:str="600px", y_name:str="Energy"):
    '''
    Html chart to visualize the ipmvp procedure 
    
    Param
    ------
    df: dataframe
        dataframe with data and datetime index
    baseline: str
        name of the baseline column
    adjusted_reporting_period: str
        name of the adjusted reporting period column
    save_mode: str
        "path":Save file in a defined path
        "embedded": Save file to be embedded in a report 
    path : str
        Path to save the HTML file.
    width_graph : str, optional
        Width of the chart (default is "1200px").
    height_graph : str, optional
        Height of the chart (default is "600px").
    y_name; str
        Nmae of the y axis

    Returns
    -------
    str
        Saves HTML file to a specific path and returns the path.
    '''
    x = df.index.strftime('%d/%m/%Y %H:%M:%S').tolist()
    lower = df[baseline].tolist()
    upper = df[adjusted_reporting_period].tolist()

    # white Area (till lower limit)
    white_area = lower  # Dal fondo fino al lower limit

    # Green area (between lower and upper)
    green_area = [u - l for u, l in zip(upper, lower)]
    green_area_modified = []
    for element in green_area:
        if element <0:
            green_area_modified.append(0)
        else:
            green_area_modified.append(element)

    # Graph
    line = (
        Line(init_opts=opts.InitOpts(width=width_graph, height=height_graph, theme=ThemeType.LIGHT))
        .add_xaxis(x)
        # White area 
        .add_yaxis(
            "",
            white_area,
            stack="total",
            areastyle_opts=opts.AreaStyleOpts(opacity=1.0, color="#FFFFFF"),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=0),
        )
        # Green area (between lower and upper)
        .add_yaxis(
            "Energy saving",
            green_area_modified,
            stack="total",
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5, color="#00C853"),
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=0),
        )
        .add_yaxis(
            "Adjusted baseline energy",
            upper,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#1B5E20")
        )
        .add_yaxis(
            "Lower Limit",
            lower,
            is_symbol_show=False,
            linestyle_opts=opts.LineStyleOpts(width=2, color="#BDBDBD")
        )
        # Visual and interactive options
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis"),  # Enable tooltips
            datazoom_opts=[opts.DataZoomOpts(type_="inside")],  # Enable zoom inside
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                orient="horizontal",
                pos_left="right",
                feature={
                    # Save as Image tool
                    "saveAsImage": opts.ToolBoxFeatureSaveAsImageOpts(
                        title="Save as Image",
                        type_="png",
                        pixel_ratio=2
                    ),
                    # Restore chart tool
                    "restore": opts.ToolBoxFeatureRestoreOpts(
                        title="Restore"
                    ),
                    # Data View tool
                    "dataView": opts.ToolBoxFeatureDataViewOpts(
                        title="Data View", 
                        is_show=True,
                        lang=["Data View", "Close", "Refresh"]
                    ),
                    
                    # Data Zoom tool
                    "dataZoom": opts.ToolBoxFeatureDataZoomOpts(
                        is_show=True,
                        zoom_title="Zoom In",back_title="Zoom Out"
                    ),
                    # Magic Type tool for switching chart types
                    "magicType": opts.ToolBoxFeatureMagicTypeOpts(
                        is_show=False,
                    ),
                }
            ),
            yaxis_opts=opts.AxisOpts(name=y_name, type_="value", position="left"),  # Primary Y-axis (left)
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        )
    )
    # Save chart as HTML
    if save_mode == "path":
        return line.render(path)
    elif save_mode == "embedded":
        return line.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")


    
    

# ================================================
#               ANALYTICAL MODEL EVALUATION
# ================================================

def scatter_actual_vs_predicted(df: pd.DataFrame, x: str, y: str, save_mode:str, width_graph: str, height_graph: str, path:str="", toolbox_option:str="Zoom"):
    '''
    Plot a scatter plot of actual values vs predicted values round a 45° slope 
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with data.
    x : str
        Name of the df column for the x-axis.
    y : str
        Name of the df column for the first y-axis.
    save_mode: str
        "path":Save file in a defined path
        "embedded": Save file to be embedded in a report 
    path : str
        Path to save the HTML file.
    width_graph : str, optional
        Width of the chart (default is "1200px").
    height_graph : str, optional
        Height of the chart (default is "600px").

    Returns
    -------
    str
        Saves HTML file to a specific path or in embedded format to be integrated in a report

    '''
    # 45-degree line
    min_val = min(df[x].min(), df[y].min())
    max_val = max(df[x].max(), df[y].max())
    line_points = [(min_val, min_val), (max_val, max_val)]

    scatter = (
        Scatter(init_opts=opts.InitOpts(width=width_graph, height=height_graph, theme=ThemeType.LIGHT))
        .add_xaxis(df[x].tolist())
        .add_yaxis(
            series_name="",
            y_axis=df[y].tolist(),
            symbol_size=5,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_series_opts()
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_="value", splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True),
            ),
            tooltip_opts=opts.TooltipOpts(is_show=False),
        )
    )

    # Add the 45-degree reference line using a Line chart
    line = (
        Line()
        .add_xaxis([pt[0] for pt in line_points])
        .add_yaxis(
            "",
            [pt[1] for pt in line_points],
            linestyle_opts=opts.LineStyleOpts(color="red", width=2),
            label_opts=opts.LabelOpts(is_show=False),
            is_symbol_show=False
        )
    )
    scatter.overlap(line)

    # Toolbox option 
    if toolbox_option == "Zoom":
        feature_toolbox={"dataZoom": {}}
    elif toolbox_option == "None":
        feature_toolbox = {}
    else:
        feature_toolbox={
            "dataZoom": {},
            "saveAsImage": {},
            "restore": {},
        }  

    # Global options
    scatter.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name = "Actual Values",
            type_="value",
            name_location="middle",
            name_gap=35,
            axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
        ),
        yaxis_opts=opts.AxisOpts(
            name = "Predicted Values",
            type_="value",
            axispointer_opts=opts.AxisPointerOpts(is_show=True, type_="shadow"),
        ),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature=feature_toolbox
        ),
    )

    # Save chart as HTML
    if save_mode == "path":
        return scatter.render(path)
    elif save_mode == "embedded":
        return scatter.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")



def residual_distribution(df:pd.DataFrame, actual_values: str, predicted_values: str, save_mode: str, path:str=None, width_graph: str="900px", height_graph: str="500px", toolbox_option:str="Zoom"):
    """
    Plots the distribution of residuals between actual and predicted values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with data.
    actual_values : str
        Name of the df column for the x-axis.
    predicted_values : str
        Name of the df column for the first y-axis.
    save_mode: str
        "path":Save file in a defined path
        "embedded": Save file to be embedded in a report 
    path : str
        Path to save the HTML file.
    width_graph : str, optional
        Width of the chart (default is "1200px").
    height_graph : str, optional
        Height of the chart (default is "600px").

    Returns
    -------
    str
        Saves HTML file to a specific path and returns the path.
    """

    # Calculate residuals
    residuals = df[actual_values] - df[predicted_values]

    # Create bins for histogram with Sturges' formula
    num_bins = int(np.ceil(np.log2(len(residuals)) + 1))  # Better bin calculation
    min_val = residuals.min()
    max_val = residuals.max()
    bin_width = (max_val - min_val) / num_bins
    bins = np.linspace(min_val, max_val, num_bins + 1)

    # Calculate histogram using numpy
    hist, _ = np.histogram(residuals, bins=bins)

    # Calculate KDE for the smooth curve
    # Use more points for a smoother KDE
    kde_x = np.linspace(min_val - bin_width, max_val + bin_width, 300)
    kde = stats.gaussian_kde(residuals)
    kde_y = kde(kde_x)

    # Scale KDE to match histogram area (not just height)
    # This ensures proper probability density representation
    area_hist = sum(hist) * bin_width
    area_kde = sum(kde_y) * (kde_x[1] - kde_x[0])
    scale_factor = area_hist / area_kde
    kde_y_scaled = kde_y * scale_factor


    # Add the KDE as a Line chart
    line = Line(init_opts=opts.InitOpts(width=width_graph, height=height_graph, theme=ThemeType.LIGHT))
    line.add_xaxis([round(x, 2) for x in kde_x])
    line.add_yaxis(
        "",
        kde_y_scaled.tolist(),
        is_smooth=True,
        symbol="none",
        color="black",
        is_symbol_show=False,
        linestyle_opts=opts.LineStyleOpts(width=2),
        areastyle_opts=opts.AreaStyleOpts(opacity=1, color="#C67570")
    )

    # Add the vertical line at x=0
    vertical_line = Line()
    vertical_line.add_xaxis([0, 0])
    vertical_line.add_yaxis(
        "",
        [0, max(hist) * 1.1],  # From 0 to just above max height
        is_symbol_show=False,
        label_opts=opts.LabelOpts(is_show=False),
        linestyle_opts=opts.LineStyleOpts(color="red", type_="dashed", width=2),
        itemstyle_opts=opts.ItemStyleOpts(color="red"),
    )

    # Overlay the charts
    line.overlap(vertical_line)

    # Toolbox option 
    if toolbox_option == "Zoom":
        feature_toolbox={"dataZoom": {}}
    elif toolbox_option == "None":
        feature_toolbox = {}
    elif toolbox_option == "All":
        feature_toolbox={
            "dataZoom": {},
            "saveAsImage": {},
            "restore": {},
        }  


    # Configure global options
    line.set_global_opts(
        # title_opts=opts.TitleOpts(
        #     title="Residuals Distribution",
        #     subtitle=f"Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f}"
        # ),
        xaxis_opts=opts.AxisOpts(
            name="Residuals (Actual - Predicted)",
            name_location="middle",
            name_gap=35,
            type_="value",
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        yaxis_opts=opts.AxisOpts(
            name="Frequency",
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis",
            axis_pointer_type="shadow",
        ),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature=feature_toolbox
        ),
    )

        # Save chart as HTML
    if save_mode == "path":
        return line.render(path)
    elif save_mode == "embedded":
        return line.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")


def residuals_vs_predicted_values(df: pd.DataFrame, save_mode: str, width_graph: str, height_graph: str, path: str, toolbox_option: str="Zoom"):
    '''
    Plot a scatter plot of residuals vs predicted values    
    '''
    # Calculate residuals
    residuals = df['actual'] - df['predicted']

    # Create data pairs for scatter plot
    data = list(zip(df['predicted'].tolist(), residuals.tolist()))

    # Create the scatter plot
    scatter = Scatter(init_opts=opts.InitOpts(width=width_graph, height=height_graph, theme=ThemeType.LIGHT))
    scatter.add_xaxis([])  # Empty xaxis as we'll use value pairs
    scatter.add_yaxis(
        series_name="Residuals",
        y_axis=data,
        symbol_size=8,
        itemstyle_opts=opts.ItemStyleOpts(
            color="blue",
            opacity=0.6,
        ),
        label_opts=opts.LabelOpts(is_show=False),
    )

    # Toolbox option 
    if toolbox_option == "Zoom":
        feature_toolbox={"dataZoom": {}}
    elif toolbox_option == "None":
        feature_toolbox = {}
    elif toolbox_option == "All":
        feature_toolbox={
            "dataZoom": {},
            "saveAsImage": {},
            "restore": {},
        }  

    # Configure global options
    scatter.set_global_opts(
        xaxis_opts=opts.AxisOpts(
            name="Predicted Values",
            name_location="middle",
            name_gap=35,
            type_="value",
            
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(
                    color="red",  # Red x-axis line
                    width=2       # Weight/width of 2
                )
            ),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        yaxis_opts=opts.AxisOpts(
            name="Residuals",
            type_="value",
            name_location="middle",
            name_gap=35,  # Increases space between axis and its name
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(
            formatter="{c}",
            trigger="item",
            axis_pointer_type="cross",
        ),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True,
            feature=feature_toolbox
        ),
        legend_opts=opts.LegendOpts(is_show=False),  # Hide the legend
    )

    # Save chart as HTML
    if save_mode == "path":
        return scatter.render(path)
    elif save_mode == "embedded":
        return scatter.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")


def features_importance(coef_df: pd.DataFrame, width_graph: str, height_graph: str, save_mode: str, path: str, toolbox_option:str="Zoom"):
    '''
    Plot a bar chart of features importance    

    Parameters
    ----------
    coef_df : pd.DataFrame
        Dataframe with features and coefficients.
    width_graph : str
        Width of the chart (default is "1200px").
    height_graph : str
        Height of the chart (default is "600px").
    save_mode : str
        "path":Save file in a defined path
        "embedded": Save file to be embedded in a report 
    path : str
        Path to save the HTML file.
    '''
    
    features = coef_df["Feature"].tolist()[::-1]  
    coefficients = coef_df["Coefficient"].tolist()[::-1]
    coefficients = [round(coef, 2) for coef in coefficients]
    
    bar = (
        Bar(init_opts=opts.InitOpts(width=width_graph, height=height_graph, theme=ThemeType.LIGHT))
        .add_xaxis(features)
        .add_yaxis("Coefficient", coefficients, label_opts=opts.LabelOpts(position="right"))
        .reversal_axis()
    )

    if toolbox_option == "Zoom":
        feature_toolbox={"dataZoom": {}}
    elif toolbox_option == "None":
        feature_toolbox = {}
    else:
        feature_toolbox={
            "dataZoom": {},
            "saveAsImage": {},
            "restore": {},
        }  

    bar.set_global_opts(
            xaxis_opts=opts.AxisOpts(
                name="Coefficient",
                name_location="middle",
                name_gap=35,
            ),
            yaxis_opts=opts.AxisOpts(
                name="Feature",
                axislabel_opts=opts.LabelOpts(
                    rotate=35,
                    font_size=10
                )
            ),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature=feature_toolbox
            )
        )


    if save_mode == "path":
        return bar.render(path)
    elif save_mode == "embedded":
        return bar.render_embed()
    else: 
        raise ValueError("Invalid save mode. Use 'path' or 'embedded'.")
    
import pandas as pd

def mean_daily_consumption_and_temperature(df:pd.DataFrame, reporting_period:list, name_energy:str, name_temperature:str, working_hours:list, year:int=None)->dict:
    '''
    Mean of daily consumption and indoor temperature for the reporting period
    Param
    ------
    df: Dataframe containing the data and index time in datetime format
    reporting_period: list of the reporting period
    name_energy: name of the energy column
    name_temperature: name of the temperature column
    working_hours: list of the working hours [start, end]

    Return 
    ------
    mean_values: dictionary containing the mean of daily consumption and indoor temperature
    '''
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['year'] = df.index.year
    # Subset df
    df_reporting = df.loc[reporting_period[0]:reporting_period[1]]
    # Calculate the mean of daily consumption and indoor temperature for workdays filtering by hours
    if year is not None:
        df_workdays = df_reporting[(df_reporting['dayofweek'] < 5) & (df_reporting['hour'] >= working_hours[0]) & (df_reporting['hour'] <= working_hours[1]) & (df_reporting['year'] == year)]
    else:
        df_workdays = df_reporting[(df_reporting['dayofweek'] < 5) & (df_reporting['hour'] >= working_hours[0]) & (df_reporting['hour'] <= working_hours[1])]
    # Calculate the mean of daily consumption and indoor temperature
    indoor_temp_mean = df_workdays.loc[:, [name_temperature]].resample('D').mean().dropna()
    mean_energy_consumption = df_workdays.loc[:, [name_energy]].resample('D').sum().dropna()
    mean_energy_consumption = mean_energy_consumption[mean_energy_consumption >0].dropna()
    # Calculate the mean of daily consumption and indoor temperature
    mean_values ={
        'daily_energy_mean': mean_energy_consumption.mean().values[0].round(2),
        'daily_temperature_mean': indoor_temp_mean.mean().values[0].round(2)
    }

    return mean_values

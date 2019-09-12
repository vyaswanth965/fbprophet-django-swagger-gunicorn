import pandas as pd
import os

base_dir='/home/vudumula/Desktop/project/predictions'
Models_dir = base_dir+'/Models'
Data_dir = base_dir+'/Data'
prediction_size = 30
min_records_to_train =900

def fill_in_missing_dates(df,idx, date_col_name , fill_value = 0):
    df.set_index(date_col_name,drop=True,inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df = df.reindex(idx,fill_value=fill_value)
    df[date_col_name] = pd.DatetimeIndex(df.index)
    return df


def readData(column_name):
    df_original = pd.DataFrame()
    if os.path.exists(Data_dir+'/'+column_name):
        for file in os.listdir(Data_dir+'/'+column_name):
            dataFrame = pd.read_csv('{}/{}/{}'.format(Data_dir,column_name,file), engine='python',keep_default_na=False,na_values=0)
            #dataFrame = pd.read_csv('{}/{}/{}'.format(Data_dir,column_name,file),engine='c')
            #dataFrame = pd.read_csv(open('{}/{}/{}'.format(Data_dir,column_name,file),'rU'), encoding='utf-8', engine='c')

            df_original = df_original.append(dataFrame,sort=False)
        if len(df_original) != 0:
            df_original['csv_scheddate'] = pd.to_datetime(df_original.csv_scheddate, format='%m/%d/%y')
            df_original['VisitCount'] = pd.to_numeric(df_original.VisitCount)

            df_original =df_original[df_original['csv_scheddate']!='8/8/19']
    return  df_original
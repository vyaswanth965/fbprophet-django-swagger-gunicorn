from sklearn.model_selection import ParameterGrid ; import itertools ; from random import sample
import pandas as pd
from fbprophet import Prophet
import numpy as np 
import pickle
import os
from datetime import datetime
import sys
from configs import base_dir,Models_dir,Data_dir,prediction_size,min_records_to_train,readData,fill_in_missing_dates



def create_CV(train_data):
    train_data = train_data.sort_values("ds").reset_index(drop = True) 
    param_grid = {'growth': ['linear','logistic'],
                'changepoint_prior_scale': list(itertools.chain(*[sample([0.01, 0.1, 0.05],1)],[0.05,0.5,2,10])),
                'changepoint_range': list(np.arange(0.65,0.9,0.05).tolist()),
                'seasonality_prior_scale': list((10**3,100,10,1,0.1,0.01)),
                'holidays_prior_scale' : list((1000,100,10,1,0.1)),
                'capacity':[(train_data.y.max())+30,(train_data.y.max())],
                'seasonality_mode':['multiplicative','additive'],
                  'd_fourier_order':list((1,5,10,15,18,20,30)),
                    'w_fourier_order':list((1,5,10,15,18,20,30)),
                    'm_fourier_order':list((1,5,10,15,18,20,30)),
                    'y_fourier_order':list((1,5,10,15,18,20,30))
                  }
    CVGrid = list(ParameterGrid(param_grid))
    i = sample(list(range(len(CVGrid))),40)
    #Using the iths indexes selected to apply a random search
    CVGrid = [CVGrid[id] for id in i]
    return(CVGrid)



def calculate_forecast_errors(df):
    df =df.copy()
    df['e'] = df['y'] - df['yhat']
    df['p'] = 100 * df['e'] / df['y']
    error_mean = lambda error_name: np.mean(np.abs(df[error_name]))
    return error_mean('p'),error_mean('e')


#def train(column_name,values):
column_name = sys.argv[1].lower()
values = sys.argv[2].upper()

df_original = readData(column_name)
if len(df_original)==0:
    print('No data available for this parameter {}'.format(column_name))
    sys.exit(1)

if str(values).upper() == 'ALL':
    s=(df_original[column_name].value_counts() > min_records_to_train)
    values=list(s[s].index)
else:
    values = values.split(',')


for value in values:
    df = df_original.loc[df_original[column_name] == value]

    prefixed = [filename for filename in os.listdir(Models_dir) if filename.startswith(value)]
    print(prefixed)
    m_date=''
    if len(prefixed)>0:
        prefixed.sort(reverse=True)
        m_date = prefixed[0].split('.')[0].split('_')[2]
    if m_date == str(df.csv_scheddate.max()):
        print('{} model upto date'.format(value))
        continue
    else:
        if len(df)>min_records_to_train:
            df = df.loc[:,['csv_scheddate','VisitCount']]
            df = df[np.abs(df.VisitCount-df.VisitCount.mean()) <= (2*df.VisitCount.std())]    
            df = df.rename(columns={'csv_scheddate': 'ds', 'VisitCount': 'y'})
            #idx = pd.date_range(df.ds.min(),df.ds.max())
            #df = fill_in_missing_dates(df,idx,'ds',df.y.mean())
            train_df = df.iloc[:-prediction_size,:]
            future = df.iloc[-prediction_size:,:]

            lst = create_CV(df)
            
            min_mape=345435
            best_param=None
            for l in lst:
                if l['growth']=='logistic':
                    train_df = train_df.assign(cap = l['capacity']) 
                    future = future.assign(cap = l['capacity'])

                m = Prophet( daily_seasonality=False,weekly_seasonality=False,\
                            yearly_seasonality=False,seasonality_mode=l['seasonality_mode']\
                            ,growth=l['growth'],changepoint_prior_scale=l['changepoint_prior_scale'],\
                        changepoint_range=l['changepoint_range'],holidays_prior_scale=l['holidays_prior_scale'],\
                        seasonality_prior_scale=l['seasonality_prior_scale'])\
                        .add_seasonality(name="monthly",period=30.5,fourier_order=l['m_fourier_order'])\
                        .add_seasonality(name='daily',period=1,fourier_order=l['m_fourier_order']).\
                        add_seasonality(name='weekly',period=7,fourier_order=l['m_fourier_order'])\
                        .add_seasonality(name='yearly',period=365.25,fourier_order=l['m_fourier_order'])
                
                m.add_country_holidays(country_name='US')

                m.fit(train_df)

                forecast = m.predict(future)

                forecast['yhat'] = np.where(forecast.yhat > 0, forecast.yhat, forecast.yhat_upper)
                forecast = pd.merge(forecast,  df,on='ds', how='left')

                MAPE,MAE = calculate_forecast_errors(forecast)
                if MAPE<min_mape:
                    min_mape=MAPE
                    min_mae =MAE
                    best_param=l
            v_capacity=0    
            if best_param['growth']=='logistic':
                df = df.assign(cap = best_param['capacity']) 
                v_capacity = best_param['capacity']

            m = Prophet( daily_seasonality=False,weekly_seasonality=False,\
                        yearly_seasonality=False,seasonality_mode=best_param['seasonality_mode']\
                        ,growth=best_param['growth'],changepoint_prior_scale=best_param['changepoint_prior_scale'],\
                    changepoint_range=best_param['changepoint_range'],holidays_prior_scale=best_param['holidays_prior_scale'],\
                    seasonality_prior_scale=best_param['seasonality_prior_scale'])\
                    .add_seasonality(name="monthly",period=30.5,fourier_order=best_param['m_fourier_order'])\
                    .add_seasonality(name='daily',period=1,fourier_order=best_param['m_fourier_order']).\
                    add_seasonality(name='weekly',period=7,fourier_order=best_param['m_fourier_order'])\
                    .add_seasonality(name='yearly',period=365.25,fourier_order=best_param['m_fourier_order'])
            
            m.add_country_holidays(country_name='US')  
            m.fit(df)
          
            filename = Models_dir+'/'+value+'_'+str(int(v_capacity))+'_'+str(df.ds.max())+'.sav'

            if len(prefixed)>0:
                os.remove(Models_dir+'/'+prefixed[0])
    
            pickle.dump(m, open(filename, 'wb')) 
            print('{} saved '.format(filename) )

        else:
            print('no sufficient records to train model for this code {}'.format(value))


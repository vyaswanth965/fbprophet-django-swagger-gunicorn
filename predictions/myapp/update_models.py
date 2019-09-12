from sklearn.model_selection import ParameterGrid ; import itertools ; from random import sample
import pandas as pd
from fbprophet import Prophet
import numpy as np 
import pickle
import os
from datetime import datetime,timedelta
import sys
from configs import base_dir,Models_dir,Data_dir,prediction_size,min_records_to_train,readData,fill_in_missing_dates

def stan_init2(m):
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    return res

column_name = sys.argv[1].lower()
values = sys.argv[2].upper()

df_original = readData(column_name)

if str(values).upper() == ['ALL']:
    s=(df_original[column_name].value_counts() > min_records_to_train)
    values=list(s[s].index)
else:
    values = values.split(',')

for value in values:
    df = df_original.loc[df_original[column_name] == value]
    print(df.csv_scheddate.max())

    prefixed = [filename for filename in os.listdir(Models_dir) if filename.startswith(value)]
    m_date=''
    if len(prefixed)>0:
        prefixed.sort(reverse=True)
        print(prefixed)
        m_date = prefixed[0].split('.')[0].split('_')[2]
        if m_date == str(df.csv_scheddate.max()):
            print('{} model upto date'.format(value))
            continue
        else:
            df = df.loc[df['csv_scheddate'] > m_date]
            df = df[np.abs(df.VisitCount-df.VisitCount.mean()) <= (2*df.VisitCount.std())]    
            df = df.rename(columns={'csv_scheddate': 'ds', 'VisitCount': 'y'})
            #idx = pd.date_range(df.ds.min(),df.ds.max())
            #df = fill_in_missing_dates(df,idx,'ds',df.y.mean())
            model2 = Prophet()
            model1 = pickle.load(open(Models_dir+'/'+prefixed[0], 'rb'))
            model2.fit(df, init=stan_init2(model1))
            cap = prefixed[0].split('_')[1]
            filename = Models_dir+'/'+value+'_'+cap+'_'+str(df.ds.max())+'.sav'
            os.remove(Models_dir+'/'+prefixed[0])   
            pickle.dump(model2, open(filename, 'wb')) 
            print('{} updated '.format(filename) )
    else:
        print('previous model not found for this code {}'.format(value))
        continue    #skipping values which are not having models




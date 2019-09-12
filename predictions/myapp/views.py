from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import JsonResponse

from .models import Medical
from rest_framework import status
import datetime

import pandas as pd
from fbprophet import Prophet
import numpy as np
import pickle
import os
from datetime import datetime
from .configs import base_dir,Models_dir,Data_dir,prediction_size,min_records_to_train,readData



@api_view(['POST'])
def get_predictions(request):
    # ----- YAML below for Swagger -----
    """
    description: This API predicts visits count based on zip_code or sc_code (date format[YYYY-MM-DD],values delimited by comma , ).
    parameters:
      - name: column_name
        type: string
        description : sc_code or zip_code
        required: true
        location: form
      - name: values
        description : sc_code or zip_code values
        type: string
        required: true
        location: form
      - name: start_date
        type: string
        description : prediction start date
        required: true
        location: form
      - name: end_date
        description : prediction end date
        type: string
        required: true
        location: form
    """
    print('1')
    column_name = request.POST.get('column_name').lower()
    values = request.POST.get('values').upper().strip(',').split(',')
    start_date = datetime.strptime(request.POST.get('start_date'), '%Y-%m-%d')
    end_date = datetime.strptime(request.POST.get('end_date'), '%Y-%m-%d')
    print('2')
    print(values)

    try:
        if start_date > end_date:
            return Response('start_date should be less than end_date')

        df_original = readData(column_name)

        if len(df_original) == 0:
            return Response('No data available for this parameter {} '.format(column_name)) 
        full_json = {}
        print('3')

        for value in values:
            df = df_original.loc[df_original[column_name] == value]
            df = df.loc[:, ['csv_scheddate', 'VisitCount']]
            future = pd.date_range(start_date, end_date).to_frame()
            future = future.rename(columns={0: 'ds'})

            prefixed = [filename for filename in os.listdir(Models_dir) if filename.startswith(value)]
            print('4')

            if len(prefixed) > 0:
                prefixed.sort(reverse=True)
                print(prefixed[0])
                cap = prefixed[0].split('_')[1]
                if cap != '0':
                    future = future.assign(cap = float(cap))

                model = pickle.load(open(Models_dir + '/' + prefixed[0], 'rb'))
            else:
                full_json[value] = 'no sufficient records to train model for this code {}'.format(value)
                continue
            print(model)
            print(future.head())
            forecast = model.predict(future)
            if len(forecast.columns)==2:
                forecast.drop('cap',axis=1,inplace=True)
            print('a')
            forecast['yhat'] = np.round(np.where(forecast.yhat > 0, forecast.yhat, forecast.yhat_upper))
            print('b')

            forecast = forecast.iloc[:, [0, 2, 3, -1]]
            print('c')

            forecast = forecast.rename(
                columns={'ds': 'date', 'yhat': 'predicted_count', 'yhat_lower': 'lower_bound', 'yhat_upper': 'upper_bound'})
            print('6')
         
            if start_date > df.csv_scheddate.max():
                forecast.fillna('NULL',inplace=True)
                js = forecast.to_dict('r')
                js = {column_name: value, 'predictions': js}
                print('7')

            else:
                print('8')

                df = df.rename(columns={'csv_scheddate': 'date', 'VisitCount': 'actual_count'})
                forecast = pd.merge(forecast, df, on='date', how='left')

                forecast['error'] = forecast['actual_count'] - forecast['predicted_count']
                forecast['error_percentage'] = 100 * forecast['error'] / forecast['actual_count']
                error_mean = lambda error_name: np.mean(np.abs(forecast[error_name]))
                MAPE = error_mean('error_percentage')
                MAE = error_mean('error')
                forecast.fillna('NULL',inplace=True)
                js = forecast.to_dict('r')
                js = {column_name: value, 'predictions': js}
                js['MAPE'] = MAPE
                js['MAE'] = MAE
            js['Confidance_interval'] = '80'
            full_json[value] = js
        print('successfull return')
        return JsonResponse(full_json, status=status.HTTP_200_OK)

    except Exception as ex:
        return Response(ex, status=status.HTTP_400_BAD_REQUEST)



        

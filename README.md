open terminal 

Run below commands

1. cd project

2. pip install -r requirements.txt

For taining models 

3. python3 predictions/myapp/train_models.py column_name values
   examples: 
   python3 predictions/myapp/train_models.py sc_code rn11h,cna11h
   python3 predictions/myapp/train_models.py zip_code 33414  
   python3 predictions/myapp/train_models.py sc_code all

4. python3 manage.py runserver
            or
   gunicorn predictions.wsgi:application

Test the project by following URL
http://127.0.0.1:8000/swagger

and give below values
column_name : sc_code
values : cna11h
start_date : 2019-7-1
end_date : 2019-8-10   

![alt text](https://github.com/vyaswanth965/fbprophet-django-swagger-gunicorn/blob/master/Screenshot%20from%202019-09-12%2016-52-52.png)
![alt text](https://github.com/vyaswanth965/fbprophet-django-swagger-gunicorn/blob/master/Screenshot%20from%202019-09-12%2016-53-36.png)
![alt text](https://github.com/vyaswanth965/fbprophet-django-swagger-gunicorn/blob/master/Screenshot%20from%202019-09-12%2016-54-29.png)


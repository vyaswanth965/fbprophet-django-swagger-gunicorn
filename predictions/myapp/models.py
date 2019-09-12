from django.db import models


class Medical(models.Model):
    column_name = models.CharField(max_length=22)
    values = models.CharField(max_length=22)
    start_date = models.CharField(max_length=44)
    end_date = models.CharField(max_length=44)    
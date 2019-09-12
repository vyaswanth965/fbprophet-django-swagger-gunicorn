from django.conf.urls import url
from .views import get_predictions




urlpatterns = [
    url(r'^get_predictions', get_predictions, name='get_predictions'),
]
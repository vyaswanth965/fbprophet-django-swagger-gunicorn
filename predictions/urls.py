from django.conf.urls import url, include
from .swagger_schema import SwaggerSchemaView

urlpatterns = [
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),
    url(r'^sample/', include('predictions.myapp.urls')),
    url(r'^swagger/', SwaggerSchemaView.as_view()),
]
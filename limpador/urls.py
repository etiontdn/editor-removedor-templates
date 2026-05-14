from django.urls import path
from . import views

urlpatterns = [
    path('', views.editor_view, name='editor'),
    path('upload/', views.upload_image, name='upload_image'),
]

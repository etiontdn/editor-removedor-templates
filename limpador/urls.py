from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('editor/', views.editor_view, name='editor'),
    path('upload/', views.upload_image, name='upload_image'),
    path('save-templates/', views.save_templates, name='save_templates'),
    path('delete-template/<int:template_id>/', views.delete_template, name='delete_template'),
]

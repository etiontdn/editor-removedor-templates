from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('editor/', views.editor_view, name='editor'),
    path('upload/', views.upload_image, name='upload_image'),
    path('save-templates/', views.save_templates, name='save_templates'),
    path('delete-template/<int:template_id>/', views.delete_template, name='delete_template'),
    path('processor/', views.processor_view, name='processor'),
    path('scan-folders/', views.scan_folders_api, name='scan_folders'),
    path('start-processing/', views.start_processing_api, name='start_processing'),
    path('progress/<str:task_id>/', views.get_progress_api, name='get_progress'),
]

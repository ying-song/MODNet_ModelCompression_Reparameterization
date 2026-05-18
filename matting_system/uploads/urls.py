from django.urls import path
from . import views

app_name = 'uploads'

urlpatterns = [
    path('upload/', views.upload_image, name='upload'),
    path('success/<int:image_id>/', views.upload_success, name='success'),
]

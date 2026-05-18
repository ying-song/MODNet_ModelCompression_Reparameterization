from django.urls import path
from . import views

app_name = 'users'

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('profile/', views.profile, name='profile'),
    path('edit-profile/', views.edit_profile, name='edit_profile'),
    path('history/', views.history, name='history'),
    path('history/delete/<int:result_id>/', views.delete_history, name='delete_history'),  # 添加删除路由
path('contact-admin/', views.contact_admin, name='contact_admin'),
    path('messages/', views.message_list, name='message_list'),
    path('messages/<int:message_id>/', views.message_detail, name='message_detail'),
    path('notices/', views.system_notices, name='system_notices'),
]

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

def home(request):
    """首页视图"""
    return render(request, 'home.html')
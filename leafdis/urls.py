from django.urls import path
from . import views
import torch

urlpatterns = [
    path('login',views.login,name='login'),
    path('signup',views.signup,name='signup'),
    path('index',views.index,name='index'),
    
    path('predictimage',views.predictimage,name='predictimage'),
    path('forum',views.forum,name='froum'),
    path('wikip',views.wikip,name='wikip'),
    path('logout', views.logout_view, name='logout'),

    ]
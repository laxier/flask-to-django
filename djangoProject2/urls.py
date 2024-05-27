"""
URL configuration for djangoProject2 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from blog import views
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index, name='index'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout, name='logout'),
    path('post/delete/<str:post_id>/', views.delete_post, name='delete_post'),
    path('info/', views.info, name='info'),
    path('neuro/gradient/study/', views.gradient_study, name='gradient_study'),
    path('neuro/gradient/', views.gradient, name='gradient'),
    path('neuro/lstm/', views.lstm, name='lstm'),
    path('neuro/lstm/study/', views.lstm_study, name='lstm_study'),
    path('neuro/rnn/', views.rnn, name='rnn'),
    path('neuro/rnn/study/', views.rnn_study, name='rnn_study'),
]

from django.shortcuts import render


def index(request):
    context = {}
    return render(request, 'index.html', context)

def login(request):
    context = {}
    return render(request, 'login.html', context)

def register(request):
    context = {}
    return render(request, 'register.html', context)

def logout(request):
    context = {}
    return render(request, 'index.html', context)


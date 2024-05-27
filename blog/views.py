from django.contrib.auth import authenticate
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib import messages
from .forms import LoginForm, RegistrationForm
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .forms import PostForm
from .models import Post
from django.conf import settings
import os

from gradient import grad_random, study_grad
from lstm import study_lstm, get_real_pericted
from rnn import predict_spam, study_rnn


@login_required
def index(request):
    if request.method == 'POST':
        form = PostForm(request.POST, request.FILES)
        if form.is_valid():
            file = form.cleaned_data['image']
            file_path = ""
            if file:
                static_path = os.path.join(settings.STATICFILES_DIRS[0], 'images')
                os.makedirs(static_path, exist_ok=True)
                # Construct the file path
                filename = os.path.join(static_path, file.name)
                if os.path.exists(filename):
                    # Create a unique file name
                    file_name, file_extension = os.path.splitext(file.name)
                    i = 1
                    new_filename = f"{file_name}_{i}{file_extension}"
                    while os.path.exists(os.path.join(settings.MEDIA_ROOT, new_filename)):
                        i += 1
                        new_filename = f"{file_name}_{i}{file_extension}"
                    filename = os.path.join(static_path, new_filename)
                else:
                    filename = os.path.join(static_path, file.name)

                print(filename)
                with open(filename, 'wb+') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)
                file_path = file.name
                messages.success(request, 'Your file has been uploaded successfully!')

            # User = get_user_model()
            # user = User.objects.get(pk=request.user.pk)
            post = Post.objects.create(
                body=form.cleaned_data['post'],
                author=request.user,
                image_path=file_path
            )
            post.save()
            messages.success(request, 'Your post is now live!')
            return redirect('index')
    else:
        form = PostForm()

    posts = Post.objects.all().order_by('-timestamp')
    return render(request, "index.html", {'form': form, 'posts': posts, 'title': 'Home Page'})


@login_required
def delete_post(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    if post:
        if request.user == post.author or request.user.is_staff:
            post.delete()
            return redirect('index')
    else:
        return redirect('index')


def info(request):
    context = {}
    return render(request, 'info.html', context)


def login(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():

            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            remember_me = form.cleaned_data['remember_me']

            user = authenticate(request, username=username, password=password)
            if user is not None:
                auth_login(request, user)
                if not remember_me:
                    request.session.set_expiry(0)  # Session expires on browser close
                return redirect('index')
            else:
                form.add_error(None, 'Invalid username or password')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


def register(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)
            messages.success(request, 'Congratulations, you are now a registered user!')
            return redirect('login')
    else:
        form = RegistrationForm()

    return render(request, 'register.html', {'form': form, 'title': 'Register'})


@login_required
def logout(request):
    auth_logout(request)
    return redirect(index)


def gradient_study(request):
    messages.info(request, study_grad(database_name='blog', collection_name='gradient'))
    return redirect('gradient')


def gradient(request):
    random_data_processed, y_random_pred = grad_random(database_name='blog', collection_name='gradient')
    if isinstance(random_data_processed, dict):
        random_data_processed = random_data_processed.items()
    return render(request, 'gradient.html', {'real': random_data_processed, 'pred': y_random_pred})


def lstm(request):
    start_date = '2015-02-11'
    end_date = '2015-03-01'
    df = get_real_pericted(start_date, end_date, 'blog', 'btc_usd')
    dict_data = df.to_dict(orient='records')
    return render(request, 'lstm.html', {'data': dict_data})


def lstm_study(request):
    study_result = study_lstm(10, 'blog', 'btc_usd')  # Assuming 1 is passed as an epoch argument
    messages.success(request, study_result)
    return redirect('lstm')


def rnn(request):
    predict = predict_spam(num_samples=10, database_name='blog', collection_name='rnn')
    return render(request, 'RNN.html', {'predict': predict})


def rnn_study(request):
    # epos = request.GET.get('epos')
    epos = 10
    study_result = study_rnn(epos, 'blog', 'rnn')
    messages.success(request, study_result)
    return redirect('rnn')

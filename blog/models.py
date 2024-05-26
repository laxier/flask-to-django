from django.db import models
from django.contrib.auth.models import AbstractUser, Group, Permission
from django.utils import timezone
from bson import ObjectId


# class User(AbstractUser):
#     # Add custom fields here if needed
#     groups = models.ManyToManyField(
#         Group,
#         related_name='blog_users',  # Unique related_name
#         blank=True,
#         help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
#         related_query_name='user',
#     )
#     user_permissions = models.ManyToManyField(
#         Permission,
#         related_name='blog_users',  # Unique related_name
#         blank=True,
#         help_text='Specific permissions for this user.',
#         related_query_name='user',
#     )


class Post(models.Model):
    # id = models.ObjectIdField(primary_key=True)
    # _id = models.ObjectIdField(primary_key=True, default=ObjectId(), editable=False)
    id = models.CharField(max_length=24, primary_key=True, default=ObjectId(), editable=False)
    body = models.CharField(max_length=140)
    timestamp = models.DateTimeField(default=timezone.now)
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE, related_name='posts')
    image_path = models.CharField(max_length=140, blank=True)

    def __str__(self):
        return f'<Post {self.body}>'

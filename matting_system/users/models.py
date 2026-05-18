from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    # 可以添加额外的用户字段
    phone = models.CharField(max_length=20, blank=True, null=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    bio = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.username

from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Forum(models.Model):
    user = models.ForeignKey(User, null=False, on_delete=models.CASCADE)
    post = models.TextField(max_length=1000, null=True,default="")
    

    def __str__(self):
        return self.user.username+self.post
from django.db import models

# Create your models here.


class UserQuery(models.Model):
    body = models.CharField(max_length = 80)
    date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.body
from django.db import models

# Create your models here.


class UserQuery(models.Model):
    body = models.CharField(max_length = 80)
    response = models.CharField(max_length = 200, blank=True, null=True)
    date = models.DateTimeField(auto_now_add=True)
    accuracy = models.DecimalField(decimal_places=3, max_digits=5, default=0.000)
    success = models.BooleanField(default=False)

    def __str__(self):
        return self.body
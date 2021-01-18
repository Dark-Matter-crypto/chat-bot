from django.contrib import admin
from .models import UserQuery

# Register your models here.

class UserQueryAdmin(admin.ModelAdmin):
    list_display = ('body', 'response', 'success', 'accuracy','date')


admin.site.register(UserQuery, UserQueryAdmin)
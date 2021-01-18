from django import forms


class UserQueryForm(forms.Form):
    question = forms.CharField(max_length=80, widget=forms.TextInput(attrs={'class': 'form-control', 'name': "question", 'style':'border-color: #243E36;'}), required=True)
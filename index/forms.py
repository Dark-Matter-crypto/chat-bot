from django import forms


class UserQueryForm(forms.Form):
    question = forms.CharField(max_length=80, widget=forms.TextInput(attrs={'class': 'form-control', 'name': "question"}), required=True)
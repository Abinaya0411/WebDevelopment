from django import forms
from .models import *


class heartForm(forms.ModelForm):
    class Meta():
        model=heartModel
        fields=['age','cp','trestbps','chol','fbs','restecg'];

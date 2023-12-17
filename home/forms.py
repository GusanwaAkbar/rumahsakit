from django import forms
from .models import PasienLama

class PasienLamaForm(forms.ModelForm):
    class Meta:
        model = PasienLama
        fields = '__all__'
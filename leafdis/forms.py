from django.forms import ModelForm,TextInput
from .models import *

class WriteBlog(ModelForm):
    class Meta:
        model = Forum
        fields = ['post'
                  ]
        widgets = {'post': TextInput({'placeholder': 'Ask Your Question Here'})}
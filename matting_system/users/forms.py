from django import forms
from django.core.exceptions import ValidationError
from .models import User  # 直接从当前应用导入自定义的User模型
from django.contrib.auth.forms import UserCreationForm

class SimpleUserCreationForm(forms.Form):
    username = forms.CharField(max_length=150, required=True)
    email = forms.CharField(required=True)
    password1 = forms.CharField(widget=forms.PasswordInput, required=True)
    password2 = forms.CharField(widget=forms.PasswordInput, required=True)

    def clean_username(self):
        username = self.cleaned_data.get('username')
        # 允许任何用户名
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        # 允许任何邮箱格式
        return email

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        # 允许任何密码
        return password1

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        # 只检查密码是否匹配
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("密码不匹配")
        return password2

    def save(self):
        username = self.cleaned_data['username']
        email = self.cleaned_data['email']
        password = self.cleaned_data['password1']

        # 创建用户 - 使用自定义的User模型
        user = User.objects.create_user(username=username, email=email, password=password)
        return user

class ProfileEditForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'phone', 'bio', 'avatar']
        widgets = {
            'email': forms.EmailInput(attrs={'class': 'form-input'}),
            'phone': forms.TextInput(attrs={'class': 'form-input'}),
            'bio': forms.Textarea(attrs={'class': 'form-input', 'rows': 3}),
        }
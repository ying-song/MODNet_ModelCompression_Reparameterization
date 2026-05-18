from django import forms
from .models import UserMessage, SystemNotice


class UserMessageForm(forms.ModelForm):
    """用户消息表单"""
    class Meta:
        model = UserMessage
        fields = ['subject', 'content', 'related_error_log']
        widgets = {
            'subject': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '请输入消息主题',
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '请详细描述您的问题或建议...',
                'rows': 6,
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
            'related_error_log': forms.Select(attrs={
                'class': 'form-control',
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
        }


class AdminReplyForm(forms.ModelForm):
    """管理员回复表单"""
    class Meta:
        model = UserMessage
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '请输入回复内容...',
                'rows': 5,
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
        }


class SystemNoticeForm(forms.ModelForm):
    """系统公告表单"""
    class Meta:
        model = SystemNotice
        fields = ['title', 'content', 'notice_type', 'related_error_log']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '公告标题',
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control',
                'placeholder': '公告内容...',
                'rows': 8,
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
            'notice_type': forms.Select(attrs={
                'class': 'form-control',
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
            'related_error_log': forms.Select(attrs={
                'class': 'form-control',
                'style': 'width: 100%; padding: 12px; border: 2px solid #e5e7eb; border-radius: 8px;'
            }),
        }

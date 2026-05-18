from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib.auth.models import User  # 导入默认的User模型
from .forms import SimpleUserCreationForm
import datetime
from .forms import ProfileEditForm

try:
    from matting.models import MattingResult, UserMessage, MattingErrorLog, SystemNotice
except ImportError:
    # 如果上面的导入失败，尝试使用绝对路径导入
    from matting_system.matting.models import MattingResult, UserMessage, MattingErrorLog, SystemNotice

try:
    from matting.message_forms import UserMessageForm, AdminReplyForm
except ImportError:
    # 如果上面的导入失败，尝试使用绝对路径导入
    from matting_system.matting.message_forms import UserMessageForm, AdminReplyForm

def register(request):
    if request.method == 'POST':
        form = SimpleUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('users:profile')
    else:
        form = SimpleUserCreationForm()
    return render(request, 'users/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('users:profile')
            else:
                messages.error(request, '用户名或密码错误。')
        else:
            messages.error(request, '用户名或密码错误')
    else:
        form = AuthenticationForm()
    return render(request, 'users/login.html', {'form': form})


def user_logout(request):
    logout(request)
    # 清除所有现有消息，避免显示之前的成功/错误消息
    storage = messages.get_messages(request)
    storage.used = True
    # 添加退出成功消息
    messages.info(request, '您已成功登出。')
    return redirect('users:login')


@login_required
def profile(request):
    """显示用户信息（只读）"""
    # 计算账户天数
    today = datetime.date.today()
    date_joined = request.user.date_joined.date()
    account_days = (today - date_joined).days + 1  # +1 是因为注册当天也算作第一天

    try:
        matting_count = MattingResult.objects.filter(user=request.user, is_deleted=False).count()
        processed_images = matting_count
    except Exception as e:
        print(f"Error counting matting results: {e}")
        # 如果字段不存在，回退到旧的查询方式
        try:
            matting_count = MattingResult.objects.filter(user=request.user).count()
            processed_images = matting_count
        except Exception as e2:
            print(f"Error with fallback counting: {e2}")
            matting_count = 0
            processed_images = 0

    return render(request, 'users/profile.html', {
        'user': request.user,
        'account_days': account_days,
        'matting_count': matting_count,
        'processed_images': processed_images
    })

@login_required
def edit_profile(request):
    """编辑用户信息"""
    if request.method == 'POST':
        print("=== POST请求调试信息 ===")
        print(f"FILES数据: {request.FILES}")
        print(f"POST数据: {request.POST}")
        if 'avatar' in request.FILES:
            print(f"上传的头像文件: {request.FILES['avatar'].name}")
            print(f"文件大小: {request.FILES['avatar'].size}")

        form = ProfileEditForm(request.POST, request.FILES, instance=request.user)
        print(f"表单是否有效: {form.is_valid()}")
        if form.is_valid():
            user = form.save()
            print(f"保存后的用户头像: {user.avatar}")
            if user.avatar:
                print(f"保存后的头像URL: {user.avatar.url}")
            messages.success(request, '个人信息更新成功！')
            return redirect('users:profile')
        else:
            # 表单验证失败，显示错误
            print("Form errors:", form.errors)
            messages.error(request, '表单验证失败，请检查输入')
    else:
        form = ProfileEditForm(instance=request.user)

    return render(request, 'users/edit_profile.html', {
        'form': form,
        'user': request.user
    })


@login_required
def history(request):
    # 获取当前用户的所有未删除抠图历史记录 - 使用 try-except 处理可能的数据库字段问题
    try:
        matting_results = MattingResult.objects.filter(user=request.user, is_deleted=False).order_by('-created_at')
    except Exception as e:
        print(f"Error fetching matting results: {e}")
        # 如果字段不存在，回退到旧的查询方式
        try:
            matting_results = MattingResult.objects.filter(user=request.user).order_by('-created_at')
        except Exception as e2:
            print(f"Error with fallback fetching: {e2}")
            matting_results = []

    return render(request, 'users/history.html', {
        'matting_results': matting_results
    })


@login_required
def delete_history(request, result_id):
    """删除历史记录（软删除）"""
    if request.method == 'POST':
        try:
            matting_result = get_object_or_404(MattingResult, id=result_id, user=request.user)
            matting_result.soft_delete()
            messages.success(request, '历史记录已成功删除。')
        except Exception as e:
            messages.error(request, f'删除记录时发生错误：{str(e)}')

    return redirect('users:history')


@login_required
def contact_admin(request):
    """联系管理员"""
    try:
        from matting.models import UserMessage, MattingErrorLog
    except ImportError:
        # 如果上面的导入失败，尝试使用绝对路径导入
        from matting_system.matting.models import UserMessage, MattingErrorLog

    try:
        from matting.message_forms import UserMessageForm
    except ImportError:
        # 如果上面的导入失败，尝试使用绝对路径导入
        from matting_system.matting.message_forms import UserMessageForm

    if request.method == 'POST':
        form = UserMessageForm(request.POST)
        if form.is_valid():
            message = form.save(commit=False)
            message.sender = request.user
            message.message_type = 'USER_TO_ADMIN'
            message.save()

            messages.success(request, '消息已发送给管理员，我们会尽快回复您！')
            return redirect('users:message_list')
    else:
        form = UserMessageForm()

    return render(request, 'users/contact_admin.html', {
        'form': form
    })


@login_required
def message_list(request):
    """消息列表"""
    try:
        from matting.models import UserMessage
    except ImportError:
        from matting_system.matting.models import UserMessage

    # 获取用户的所有消息（包括发送和接收的）
    messages = UserMessage.get_user_messages(request.user)

    # 按创建时间倒序排列
    messages = messages.order_by('-created_at')

    # 计算未读消息数量
    # 未读消息定义：
    # 1. 用户作为接收者的 UNREAD 消息
    # 2. 用户发送给用户到管理员的消息，其回复（ADMIN_TO_USER）且状态为 READ 但用户尚未查看的消息
    # 注意：这里简化逻辑，只统计用户作为接收者的未读消息
    unread_count = UserMessage.objects.filter(
        recipient=request.user,
        status='UNREAD'
    ).count()

    return render(request, 'users/message_list.html', {
        'messages': messages,
        'unread_count': unread_count
    })


@login_required
def message_detail(request, message_id):
    try:
        from matting.models import MattingResult, UserMessage, MattingErrorLog, SystemNotice
    except ImportError:
        from matting_system.matting.models import MattingResult, UserMessage, MattingErrorLog, SystemNotice

    try:
        from matting.message_forms import UserMessageForm, AdminReplyForm
    except ImportError:
        from matting_system.matting.message_forms import UserMessageForm, AdminReplyForm

    """消息详情"""
    message = get_object_or_404(UserMessage, id=message_id)

    if message.sender != request.user and message.recipient != request.user:
        messages.error(request, '无权查看此消息')
        return redirect('users:message_list')

    if message.status == 'UNREAD' and message.recipient == request.user:
        message.mark_as_read()

    root_message = message
    while root_message.replied_message:
        root_message = root_message.replied_message

    replies = root_message.replies.all().order_by('created_at')

    # 处理所有用户的回复（包括管理员和普通用户）
    if request.method == 'POST':
        reply_content = request.POST.get('reply_content', '').strip()
        if reply_content:
            # 判断回复目标
            if root_message.message_type == 'USER_TO_ADMIN':
                # 如果是用户发给管理员的消息，回复给管理员
                recipient = root_message.recipient
                message_type = 'USER_TO_ADMIN' if not request.user.is_staff else 'ADMIN_TO_USER'
            else:
                # 如果是管理员发给用户的消息，回复给用户
                recipient = root_message.sender
                message_type = 'USER_TO_ADMIN' if not request.user.is_staff else 'ADMIN_TO_USER'

            UserMessage.objects.create(
                sender=request.user,
                recipient=recipient,
                message_type=message_type,
                subject=f"Re: {root_message.subject}",
                content=reply_content,
                replied_message=root_message,
                related_error_log=root_message.related_error_log,
                status='UNREAD'
            )
            messages.success(request, '回复发送成功！')
            return redirect('users:message_detail', message_id=root_message.id)

    # 如果是管理员且需要回复（保留原来的表单逻辑作为备用）
    reply_form = None
    if request.user.is_staff and message.message_type == 'USER_TO_ADMIN':
        if request.method == 'POST' and not request.POST.get('reply_content'):
            reply_form = AdminReplyForm(request.POST)
            if reply_form.is_valid():
                reply = reply_form.save(commit=False)
                reply.sender = request.user
                reply.recipient = message.sender
                reply.message_type = 'ADMIN_TO_USER'
                reply.replied_message = message
                reply.related_error_log = message.related_error_log
                reply.status = 'UNREAD'
                reply.save()

                message.status = 'REPLIED'
                message.save()

                messages.success(request, '回复成功！')
                return redirect('users:message_detail', message_id=message.id)
        else:
            reply_form = AdminReplyForm()

    return render(request, 'users/message_detail.html', {
        'message': message,
        'root_message': root_message,
        'reply_form': reply_form,
        'replies': replies
    })


@login_required
def system_notices(request):
    """系统公告列表"""
    notices = SystemNotice.get_published_notices()

    return render(request, 'users/system_notices.html', {
        'notices': notices
    })

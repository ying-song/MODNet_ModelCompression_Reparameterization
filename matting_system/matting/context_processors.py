from django.db.models import Q

try:
    from .models import UserMessage
except ImportError:
    from matting_system.matting.models import UserMessage


def unread_messages(request):
    """
    上下文处理器：为所有模板提供未读消息数量
    """
    if not request.user.is_authenticated:
        return {'unread_messages_count': 0}

    # 统计用户作为接收者的未读消息
    unread_count = UserMessage.objects.filter(
        recipient=request.user,
        status='UNREAD'
    ).count()

    return {'unread_messages_count': unread_count}

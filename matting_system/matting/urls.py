from django.urls import path
from . import views
from django.http import JsonResponse
from django.contrib.admin.views.decorators import staff_member_required
from django.utils import timezone
from .models import MattingErrorLog, SystemNotice
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.shortcuts import redirect
import json

app_name = 'matting'

def submit_notice_view(request):
    """处理公告提交"""
    if request.method == 'POST':
        log_ids = request.POST.get('log_ids', '')
        notice_title = request.POST.get('notice_title', '').strip()
        notice_content = request.POST.get('notice_content', '').strip()

        if not notice_content:
            messages.error(request, '公告内容不能为空。')
            return redirect('/admin/matting/mattingerrorlog/')

        if not log_ids:
            messages.error(request, '请选择错误日志。')
            return redirect('/admin/matting/mattingerrorlog/')

        error_logs = MattingErrorLog.objects.filter(id__in=log_ids.split(','), resolved=True)

        category_title_map = {
            'MODEL_LOAD': '模型加载',
            'IMAGE_PROCESS': '图像处理',
            'PREDICTION': '模型预测',
            'FILE_SAVE': '文件保存',
            'OTHER': '其他问题',
        }

        notice_count = 0
        for error_log in error_logs:
            if notice_title:
                title = notice_title
            else:
                title = f"{category_title_map.get(error_log.error_category, '系统')}问题已修复"

            SystemNotice.objects.create(
                title=title,
                content=notice_content,
                notice_type='ERROR_RESOLVED',
                related_error_log=error_log,
                published_by=request.user,
                is_published=True,
                published_at=timezone.now()
            )
            error_log.notify_users = True
            error_log.save()
            notice_count += 1

        messages.success(request, f'✅ 成功为 {notice_count} 条错误日志发布系统公告。')
        return redirect('/admin/matting/mattingerrorlog/')

    return redirect('/admin/matting/mattingerrorlog/')


def send_reply_view(request):
    """发送回复消息（AJAX）"""
    from .models import UserMessage
    import json

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message_id = data.get('message_id')
            content = data.get('content', '').strip()

            if not content:
                return JsonResponse({'success': False, 'message': '回复内容不能为空。'})

            original_message = UserMessage.objects.get(pk=message_id)

            reply = UserMessage.objects.create(
                sender=request.user,
                recipient=original_message.sender,
                message_type='ADMIN_TO_USER',
                content=content,
                replied_message=original_message,
                related_error_log=original_message.related_error_log,
                status='UNREAD'
            )

            original_message.status = 'REPLIED'
            original_message.save()

            return JsonResponse({
                'success': True,
                'message': '发送成功',
                'reply_id': reply.id
            })
        except UserMessage.DoesNotExist:
            return JsonResponse({'success': False, 'message': '消息不存在。'})
        except Exception as e:
            return JsonResponse({'success': False, 'message': f'发送失败：{str(e)}'})

    return JsonResponse({'success': False, 'message': '无效的请求方法。'})


urlpatterns = [
    path('generate/<int:image_id>/', views.generate_matting, name='generate'),
    path('result/<int:result_id>/', views.result, name='result'),
    path('change-background/<int:result_id>/', views.change_background, name='change_background'),
    path('admin/mark-log-resolved/<int:log_id>/',
         staff_member_required(lambda request, log_id: mark_log_resolved(request, log_id)), name='mark_log_resolved'),
    path('submit-notice/',
         staff_member_required(csrf_exempt(submit_notice_view)),
         name='submit_notice'),
    path('send-reply/',
         staff_member_required(csrf_exempt(send_reply_view)),
         name='send_reply'),
]

def mark_log_resolved(request, log_id):
    """快速标记错误日志为已解决"""
    try:
        error_log = MattingErrorLog.objects.get(pk=log_id)
        error_log.resolved = True
        error_log.resolved_at = timezone.now()
        error_log.resolved_by = request.user
        error_log.save()

        return JsonResponse({
            'success': True,
            'message': f'日志 #{log_id} 已标记为已解决'
        })
    except MattingErrorLog.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': '日志不存在'
        }, status=404)


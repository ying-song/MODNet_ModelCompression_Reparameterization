from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import MattingResult, MattingErrorLog, UserMessage, SystemNotice
from django import forms

from django import forms


class ReplyInlineForm(forms.ModelForm):
    """回复表单"""

    class Meta:
        model = UserMessage
        fields = ['content']


class ReplyInline(admin.TabularInline):
    """回复内联编辑器"""
    model = UserMessage
    form = ReplyInlineForm
    extra = 1
    fields = ('content',)
    verbose_name = '回复'
    verbose_name_plural = '回复记录'

    def save_formset(self, request, form, formset, change):
        """保存表单集时自动设置发送者和相关字段"""
        # 获取未保存的实例
        instances = formset.save(commit=False)

        # 先保存回复
        for instance in instances:
            # 自动设置发送者为当前管理员
            instance.sender = request.user
            # 自动设置为管理员回复用户的消息类型
            instance.message_type = 'ADMIN_TO_USER'
            # 自动设置接收者为原消息的发送者
            instance.recipient = form.instance.sender
            # 关联到原消息
            instance.replied_message = form.instance
            # 关联到相关错误日志
            instance.related_error_log = form.instance.related_error_log
            # 管理员回复用户的消息，对用户来说是未读的
            instance.status = 'UNREAD'
            instance.save()

        # 处理多对多关系
        formset.save_m2m()

        # 重要：重新获取原消息实例，确保数据是最新的
        original_message = UserMessage.objects.get(pk=form.instance.pk)

        # 更新原消息状态为已回复（管理员已处理）
        if original_message.message_type == 'USER_TO_ADMIN':
            original_message.status = 'REPLIED'
            original_message.save()

    def has_add_permission(self, request, obj=None):
        # 只有在查看已有消息时才允许添加回复
        if obj and obj.message_type == 'USER_TO_ADMIN':
            return True
        return False

    def has_delete_permission(self, request, obj=None):
        return False

@admin.register(UserMessage)
class UserMessageAdmin(admin.ModelAdmin):
    list_display = (
        'message_type_badge',
        'subject_preview',
        'sender_link',
        'recipient_link',
        'status_badge',
        'created_at',
        'has_replies',
    )
    list_filter = ('message_type', 'status', 'created_at')
    search_fields = ('subject', 'content', 'sender__username', 'recipient__username')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

    fieldsets = (
        ('基本信息', {
            'fields': ('message_type', 'sender', 'recipient', 'subject', 'content'),
            'classes': ('collapse open',)
        }),
        ('状态信息', {
            'fields': ('status', 'created_at', 'read_at', 'replied_message'),
            'classes': ('collapse',)
        }),
        ('关联信息', {
            'fields': ('related_error_log',),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('created_at', 'read_at', 'replied_message', 'sender', 'recipient', 'message_type')

    # 添加内联回复
    inlines = [ReplyInline]

    actions = ['reply_to_selected', 'batch_reply_to_selected', 'mark_as_read', 'mark_as_replied']

    class Media:
        css = {
            'all': (
                'https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css',
            )
        }

    def message_type_badge(self, obj):
        """消息类型徽章"""
        if obj.message_type == 'USER_TO_ADMIN':
            return format_html(
                '<span style="display: inline-flex; align-items: center; gap: 6px; '
                'padding: 4px 12px; border-radius: 12px; '
                'background: #3b82f6; color: white; font-weight: 600; font-size: 0.85rem;">'
                '<i class="fas fa-user"></i>'
                '<span>用户→管理员</span>'
                '</span>'
            )
        else:
            return format_html(
                '<span style="display: inline-flex; align-items: center; gap: 6px; '
                'padding: 4px 12px; border-radius: 12px; '
                'background: #8b5cf6; color: white; font-weight: 600; font-size: 0.85rem;">'
                '<i class="fas fa-user-shield"></i>'
                '<span>管理员→用户</span>'
                '</span>'
            )

    message_type_badge.short_description = '类型'

    def subject_preview(self, obj):
        """主题预览"""
        preview = obj.subject[:50] + '...' if len(obj.subject) > 50 else obj.subject
        return format_html(
            '<span style="color: #1a1f36; font-weight: 600;">{}</span>',
            preview
        )

    subject_preview.short_description = '主题'

    def sender_link(self, obj):
        """发送者链接"""
        # 如果是管理员回复用户的消息，发送者显示为"管理员"
        if obj.message_type == 'ADMIN_TO_USER':
            return format_html('<span style="color: #4a6cf7; font-weight: 600;">管理员</span>')

        if obj.sender:
            url = reverse('admin:users_user_change', args=[obj.sender.id])
            return format_html('<a href="{}">{}</a>', url, obj.sender.username)
        return '-'

    sender_link.short_description = '发送者'

    def recipient_link(self, obj):
        """接收者链接"""
        # 如果是用户发给管理员的消息，接收者显示为"管理员"
        if obj.message_type == 'USER_TO_ADMIN':
            return format_html('<span style="color: #4a6cf7; font-weight: 600;">管理员</span>')

        if obj.recipient:
            url = reverse('admin:users_user_change', args=[obj.recipient.id])
            return format_html('<a href="{}">{}</a>', url, obj.recipient.username)
        return format_html('<span style="color: #9ca3af;">-</span>')

    recipient_link.short_description = '接收者'

    def status_badge(self, obj):
        """状态徽章"""
        status_config = {
            'UNREAD': ('#ef4444', 'fa-eye-slash', '未读'),
            'READ': ('#3b82f6', 'fa-check', '已读'),
            'REPLIED': ('#10b981', 'fa-reply', '已回复'),
        }
        color, icon, label = status_config.get(obj.status, ('#6b7280', 'fa-circle', obj.status))

        return format_html(
            '<span style="display: inline-flex; align-items: center; gap: 6px; '
            'color: {}; font-weight: 600;">'
            '<i class="fas {}"></i>'
            '<span>{}</span>'
            '</span>',
            color, icon, label
        )

    status_badge.short_description = '状态'

    def has_replies(self, obj):
        """是否有回复"""
        reply_count = obj.replies.count()
        if reply_count > 0:
            return format_html(
                '<span style="color: #10b981; font-weight: 600;">'
                '<i class="fas fa-comments"></i> {} 条回复'
                '</span>',
                reply_count
            )
        return format_html(
            '<span style="color: #9ca3af;">'
            '<i class="fas fa-comment-slash"></i> 无回复'
            '</span>'
        )

    has_replies.short_description = '回复情况'

    def reply_to_selected(self, request, queryset):
        """回复选中的用户消息 - 打开聊天窗口（显示完整对话历史）"""
        from django.http import HttpResponse

        selected_count = queryset.filter(message_type='USER_TO_ADMIN').count()
        if selected_count != 1:
            self.message_user(request, '请选择一条用户消息进行回复', level='error')
            return

        message = queryset.filter(message_type='USER_TO_ADMIN').first()

        # 找到根消息（初始消息）
        root_message = message
        while root_message.replied_message:
            root_message = root_message.replied_message

        # 获取所有对话历史
        all_messages = [root_message]
        all_messages.extend(root_message.replies.all().order_by('created_at'))

        message_id = message.id
        username = message.sender.username if message.sender else '未知用户'
        subject = root_message.subject
        created_at = root_message.created_at.strftime('%Y-%m-%d %H:%M')

        # 构建对话消息 HTML
        messages_html = ''
        for msg in all_messages:
            if msg.message_type == 'USER_TO_ADMIN':
                messages_html += f'''
                    <div class="message message-user">
                        <div>{msg.content}</div>
                        <div class="message-meta"><i class="fas fa-user"></i> {msg.sender.username} (用户) · {msg.created_at.strftime('%Y-%m-%d %H:%M')}</div>
                    </div>
                    '''
            else:
                messages_html += f'''
                    <div class="message message-admin">
                        <div>{msg.content}</div>
                        <div class="message-meta"><i class="fas fa-user-shield"></i> {msg.sender.username} (管理员) · {msg.created_at.strftime('%Y-%m-%d %H:%M')}</div>
                    </div>
                    '''

        html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>与 {username} 对话</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        background: #f5f5f5;
                        height: 100vh;
                        display: flex;
                        flex-direction: column;
                    }}
                    .chat-header {{
                        background: linear-gradient(135deg, #4a6cf7 0%, #6b8aff 100%);
                        color: white;
                        padding: 1rem 2rem;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    }}
                    .chat-header h2 {{
                        font-size: 1.3rem;
                        margin-bottom: 0.3rem;
                    }}
                    .chat-header p {{
                        font-size: 0.9rem;
                        opacity: 0.9;
                    }}
                    .chat-messages {{
                        flex: 1;
                        overflow-y: auto;
                        padding: 2rem;
                        display: flex;
                        flex-direction: column;
                        gap: 1rem;
                    }}
                    .message {{
                        max-width: 70%;
                        padding: 1rem 1.5rem;
                        border-radius: 12px;
                        line-height: 1.6;
                        position: relative;
                    }}
                    .message-user {{
                        align-self: flex-start;
                        background: white;
                        border: 2px solid #e5e7eb;
                        color: #1a1f36;
                    }}
                    .message-admin {{
                        align-self: flex-end;
                        background: linear-gradient(135deg, #4a6cf7 0%, #6b8aff 100%);
                        color: white;
                    }}
                    .message-meta {{
                        font-size: 0.8rem;
                        margin-top: 0.5rem;
                        opacity: 0.7;
                    }}
                    .chat-input {{
                        background: white;
                        padding: 1.5rem 2rem;
                        border-top: 2px solid #e5e7eb;
                        display: flex;
                        gap: 1rem;
                        align-items: flex-end;
                    }}
                    .chat-input textarea {{
                        flex: 1;
                        padding: 1rem;
                        border: 2px solid #e5e7eb;
                        border-radius: 12px;
                        resize: none;
                        font-size: 1rem;
                        min-height: 60px;
                        max-height: 150px;
                    }}
                    .chat-input textarea:focus {{
                        outline: none;
                        border-color: #4a6cf7;
                    }}
                    .btn-send {{
                        padding: 1rem 2rem;
                        background: linear-gradient(135deg, #4a6cf7 0%, #6b8aff 100%);
                        color: white;
                        border: none;
                        border-radius: 12px;
                        font-size: 1rem;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.3s;
                    }}
                    .btn-send:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.3);
                    }}
                    .btn-back {{
                        position: absolute;
                        top: 1rem;
                        right: 2rem;
                        background: rgba(255,255,255,0.2);
                        color: white;
                        padding: 0.5rem 1rem;
                        border-radius: 8px;
                        text-decoration: none;
                        font-size: 0.9rem;
                    }}
                    .btn-back:hover {{
                        background: rgba(255,255,255,0.3);
                    }}
                </style>
            </head>
            <body>
                <div class="chat-header" style="position: relative;">
                    <h2><i class="fas fa-comments"></i> 与 {username} 对话</h2>
                    <p>主题：{subject} | 时间：{created_at}</p>
                    <a href="/admin/matting/usermessage/" class="btn-back"><i class="fas fa-arrow-left"></i> 返回列表</a>
                </div>

                <div class="chat-messages" id="chatMessages">
                    {messages_html}
                </div>

                <div class="chat-input">
                    <textarea id="messageInput" placeholder="输入回复内容...（按 Enter 发送，Shift+Enter 换行）"></textarea>
                    <button class="btn-send" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i> 发送
                    </button>
                </div>

                <script>
                    const messageInput = document.getElementById('messageInput');
                    const chatMessages = document.getElementById('chatMessages');
                    const rootMessageId = {root_message.id};

                    messageInput.addEventListener('keydown', function(e) {{
                        if (e.key === 'Enter' && !e.shiftKey) {{
                            e.preventDefault();
                            sendMessage();
                        }}
                    }});

                    function sendMessage() {{
                        const content = messageInput.value.trim();
                        if (!content) return;

                        fetch('/matting/send-reply/', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json',
                                'X-CSRFToken': getCookie('csrftoken')
                            }},
                            body: JSON.stringify({{
                                message_id: rootMessageId,
                                content: content
                            }})
                        }})
                        .then(response => response.json())
                        .then(data => {{
                            if (data.success) {{
                                appendMessage(content, 'admin');
                                messageInput.value = '';
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }} else {{
                                alert('发送失败：' + data.message);
                            }}
                        }})
                        .catch(error => {{
                            alert('网络错误：' + error);
                        }});
                    }}

                    function appendMessage(content, type) {{
                        const now = new Date().toLocaleString('zh-CN');
                        const messageDiv = document.createElement('div');
                        messageDiv.className = `message message-${{type}}`;
                        messageDiv.innerHTML = `
                            <div>${{content}}</div>
                            <div class="message-meta">
                                <i class="fas fa-${{type === 'admin' ? 'user-shield' : 'user'}}"></i> 
                                ${{type === 'admin' ? '管理员' : '{username}'}} · ${{now}}
                            </div>
                        `;
                        chatMessages.appendChild(messageDiv);
                    }}

                    function getCookie(name) {{
                        let cookieValue = null;
                        if (document.cookie && document.cookie !== '') {{
                            const cookies = document.cookie.split(';');
                            for (let i = 0; i < cookies.length; i++) {{
                                const cookie = cookies[i].trim();
                                if (cookie.substring(0, name.length + 1) === (name + '=')) {{
                                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                                    break;
                                }}
                            }}
                        }}
                        return cookieValue;
                    }}

                    chatMessages.scrollTop = chatMessages.scrollHeight;
                    messageInput.focus();
                </script>
            </body>
            </html>
            """
        return HttpResponse(html)

    reply_to_selected.short_description = "💬 回复选中的用户消息"


    def batch_reply_to_selected(self, request, queryset):
        """批量回复选中的用户消息"""
        from django import forms
        from .message_forms import UserMessageForm

        # 检查是否选择了消息
        selected_count = queryset.filter(message_type='USER_TO_ADMIN').count()
        if selected_count == 0:
            self.message_user(request, '请选择至少一条用户消息进行批量回复', level='error')
            return

        # 如果只选择了一条，直接跳转到回复页面
        if selected_count == 1:
            message = queryset.filter(message_type='USER_TO_ADMIN').first()
            if message.replies.exists():
                self.message_user(request, '该消息已有回复，请在详情页查看', level='warning')
                return

            # 创建默认回复
            reply = UserMessage.objects.create(
                sender=request.user,
                recipient=message.sender,
                message_type='ADMIN_TO_USER',
                content='您好，我们已经收到您的问题，会尽快处理并回复您。',
                replied_message=message,
                related_error_log=message.related_error_log,
                status='UNREAD'  # 对用户来说是未读的
            )
            message.status = 'REPLIED'
            message.save()
            self.message_user(request, f'成功回复了用户 "{message.sender.username}" 的消息。')
            return

        # 批量回复多条消息
        replied_count = 0
        for message in queryset.filter(message_type='USER_TO_ADMIN'):
            if message.replies.exists():
                continue

            # 为每条消息创建相同的回复
            reply = UserMessage.objects.create(
                sender=request.user,
                recipient=message.sender,
                message_type='ADMIN_TO_USER',
                content='您好，我们已经收到您的问题，会尽快处理并回复您。',
                replied_message=message,
                related_error_log=message.related_error_log,
                status='UNREAD'  # 对用户来说是未读的
            )
            message.status = 'REPLIED'
            message.save()
            replied_count += 1

        self.message_user(request, f'成功批量回复了 {replied_count} 条用户消息。')

    batch_reply_to_selected.short_description = "📬 批量回复选中的用户消息"


    def mark_as_read(self, request, queryset):
        """批量标记为已读"""
        updated = queryset.update(status='READ')
        self.message_user(request, f'成功标记了 {updated} 条消息为已读。')

    mark_as_read.short_description = "标记选中的消息为已读"

    def mark_as_replied(self, request, queryset):
        """批量标记为已回复"""
        updated = queryset.update(status='REPLIED')
        self.message_user(request, f'成功标记了 {updated} 条消息为已回复。')

    mark_as_replied.short_description = "标记选中的消息为已回复"

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('sender', 'recipient', 'replied_message', 'related_error_log').prefetch_related(
            'replies')

@admin.register(SystemNotice)
class SystemNoticeAdmin(admin.ModelAdmin):
    list_display = (
        'notice_type_badge',
        'title_preview',
        'is_published_badge',
        'published_by_link',
        'published_at',
    )
    list_filter = ('notice_type', 'is_published', 'published_at')
    search_fields = ('title', 'content')
    date_hierarchy = 'published_at'
    ordering = ('-published_at',)

    fieldsets = (
        ('公告信息', {
            'fields': ('title', 'content', 'notice_type'),
            'classes': ('collapse open',)
        }),
        ('发布设置', {
            'fields': ('is_published', 'published_at', 'published_by', 'related_error_log'),
            'classes': ('collapse',)
        }),
    )

    readonly_fields = ('published_at', 'published_by', 'created_at', 'updated_at')

    actions = ['publish_notices', 'unpublish_notices']

    def notice_type_badge(self, obj):
        """公告类型徽章"""
        type_config = {
            'GENERAL': ('#3b82f6', 'fa-bullhorn', '一般公告'),
            'ERROR_RESOLVED': ('#10b981', 'fa-check-circle', '问题已解决'),
            'SYSTEM_MAINTENANCE': ('#f59e0b', 'fa-tools', '系统维护'),
            'FEATURE_UPDATE': ('#8b5cf6', 'fa-rocket', '功能更新'),
        }
        color, icon, label = type_config.get(obj.notice_type, ('#6b7280', 'fa-circle', obj.notice_type))

        return format_html(
            '<span style="display: inline-flex; align-items: center; gap: 6px; '
            'padding: 4px 12px; border-radius: 12px; '
            'background: {}; color: white; font-weight: 600; font-size: 0.85rem;">'
            '<i class="fas {}"></i>'
            '<span>{}</span>'
            '</span>',
            color, icon, label
        )

    notice_type_badge.short_description = '类型'

    def title_preview(self, obj):
        """标题预览"""
        preview = obj.title[:50] + '...' if len(obj.title) > 50 else obj.title
        return format_html(
            '<span style="color: #1a1f36; font-weight: 600;">{}</span>',
            preview
        )

    title_preview.short_description = '标题'

    def is_published_badge(self, obj):
        """发布状态徽章"""
        if obj.is_published:
            return format_html(
                '<span style="color: #10b981; font-weight: 600;">'
                '<i class="fas fa-check-circle"></i> 已发布'
                '</span>'
            )
        else:
            return format_html(
                '<span style="color: #ef4444; font-weight: 600;">'
                '<i class="fas fa-clock"></i> 未发布'
                '</span>'
            )

    is_published_badge.short_description = '状态'

    def published_by_link(self, obj):
        """发布者链接"""
        if obj.published_by:
            url = reverse('admin:users_user_change', args=[obj.published_by.id])
            return format_html('<a href="{}">{}</a>', url, obj.published_by.username)
        return '-'

    published_by_link.short_description = '发布者'

    def publish_notices(self, request, queryset):
        """批量发布公告"""
        from django.utils import timezone
        for notice in queryset.filter(is_published=False):
            notice.publish()
        self.message_user(request, f'成功发布了 {queryset.filter(is_published=False).count()} 条公告。')

    publish_notices.short_description = "发布选中的公告"

    def unpublish_notices(self, request, queryset):
        """批量取消发布公告"""
        updated = queryset.update(is_published=False)
        self.message_user(request, f'成功取消了 {updated} 条公告的发布。')

    unpublish_notices.short_description = "取消发布选中的公告"

    def save_model(self, request, obj, form, change):
        """保存模型时自动设置发布者"""
        if not change and not obj.published_by:
            obj.published_by = request.user
        super().save_model(request, obj, form, change)

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('published_by', 'related_error_log')


@admin.register(MattingErrorLog)
class MattingErrorLogAdmin(admin.ModelAdmin):
    list_display = (
        'error_level_badge',
        'error_category_badge',
        'error_message_preview',
        'user',
        'model_type',
        'created_at',
        'resolved_status',
    )
    list_filter = ('error_level', 'error_category', 'resolved', 'created_at', 'user')
    search_fields = ('error_message', 'error_traceback', 'user__username', 'image_path')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

    fieldsets = (
        ('基本信息', {
            'fields': ('error_level', 'error_category', 'error_message', 'created_at'),
            'classes': ('collapse open',)
        }),
        ('详细信息', {
            'fields': ('user', 'image_path', 'model_type', 'request_params'),
            'classes': ('collapse',)
        }),
        ('错误堆栈', {
            'fields': ('error_traceback',),
            'classes': ('collapse', 'monospace'),
            'description': '📋 完整的错误堆栈信息'
        }),
        ('解决状态', {
            'fields': ('resolved', 'resolved_at', 'resolved_by', 'notes', 'notify_users', 'public_notice'),
            'classes': ('collapse',),
            'description': '✅ 标记问题已解决，可选择是否通知所有用户'
        }),
    )

    readonly_fields = ('created_at', 'resolved_at', 'resolved_by', 'error_traceback')

    actions = ['mark_as_resolved', 'publish_fix_notice', 'export_selected_errors']

    class Media:
        css = {
            'all': (
                'https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css',
            )
        }
        js = ('https://cdn.jsdelivr.net/npm/sweetalert2@11',)

    def error_level_badge(self, obj):
        """错误级别徽章"""
        level_colors = {
            'INFO': '#3b82f6',
            'WARNING': '#f59e0b',
            'ERROR': '#ef4444',
            'CRITICAL': '#dc2626',
        }
        level_icons = {
            'INFO': 'fa-info-circle',
            'WARNING': 'fa-exclamation-triangle',
            'ERROR': 'fa-times-circle',
            'CRITICAL': 'fa-bug',
        }

        color = level_colors.get(obj.error_level, '#6b7280')
        icon = level_icons.get(obj.error_level, 'fa-circle')

        return format_html(
            '<span style="display: inline-flex; align-items: center; gap: 6px; '
            'padding: 4px 12px; border-radius: 12px; '
            'background: {}; color: white; font-weight: 600; font-size: 0.85rem;">'
            '<i class="fas {}"></i>'
            '<span>{}</span>'
            '</span>',
            color,
            icon,
            obj.error_level
        )

    error_level_badge.short_description = '级别'

    def error_category_badge(self, obj):
        """错误类别徽章"""
        category_icons = {
            'MODEL_LOAD': 'fa-weight-hanging',
            'IMAGE_PROCESS': 'fa-image',
            'PREDICTION': 'fa-brain',
            'FILE_SAVE': 'fa-save',
            'OTHER': 'fa-question-circle',
        }

        icon = category_icons.get(obj.error_category, 'fa-question-circle')

        return format_html(
            '<span style="display: inline-flex; align-items: center; gap: 6px; '
            'color: #6b7280; font-size: 0.9rem;">'
            '<i class="fas {}"></i>'
            '<span>{}</span>'
            '</span>',
            icon,
            obj.get_error_category_display()
        )

    error_category_badge.short_description = '类别'

    def error_message_preview(self, obj):
        """错误信息预览"""
        message = obj.error_message[:100] + '...' if len(obj.error_message) > 100 else obj.error_message
        return format_html(
            '<span style="color: #ef4444; font-family: monospace; font-size: 0.85rem;">{}</span>',
            message
        )

    error_message_preview.short_description = '错误信息'

    def resolved_status(self, obj):
        """解决状态"""
        if obj.resolved:
            return format_html(
                '<span style="color: #10b981; font-weight: 600;">'
                '<i class="fas fa-check-circle"></i> 已解决'
                '</span>'
            )
        else:
            return format_html(
                '<span style="color: #ef4444; font-weight: 600;">'
                '<i class="fas fa-times-circle"></i> 未解决'
                '</span>'
            )

    resolved_status.short_description = '状态'

    def mark_as_resolved(self, request, queryset):
        """标记选中的日志为已解决"""
        from django.utils import timezone

        resolved_count = 0
        for error_log in queryset:
            if not error_log.resolved:
                error_log.resolved = True
                error_log.resolved_at = timezone.now()
                error_log.resolved_by = request.user
                error_log.save()
                resolved_count += 1

        self.message_user(request, f'✅ 成功解决 {resolved_count} 条日志。')

    mark_as_resolved.short_description = "✅ 解决问题"

    def publish_fix_notice(self, request, queryset):
        """为已解决的问题发布公告 - 使用 JavaScript 弹窗输入内容"""
        from django.utils import timezone
        from django.contrib.admin.helpers import ACTION_CHECKBOX_NAME
        from django.http import HttpResponse

        unresolved = queryset.filter(resolved=False)
        if unresolved.exists():
            self.message_user(request, f'⚠️ 有 {unresolved.count()} 条日志尚未解决，请先解决后再发布公告。',
                              level='warning')
            return

        selected = request.POST.getlist(ACTION_CHECKBOX_NAME)
        if not selected:
            self.message_user(request, '请选择要发布公告的错误日志。', level='warning')
            return

        ids = ','.join(selected)

        error_logs = MattingErrorLog.objects.filter(id__in=selected)

        category_title_map = {
            'MODEL_LOAD': '模型加载',
            'IMAGE_PROCESS': '图像处理',
            'PREDICTION': '模型预测',
            'FILE_SAVE': '文件保存',
            'OTHER': '其他问题',
        }

        categories = set()
        for log in error_logs:
            categories.add(log.error_category)

        if len(categories) == 1:
            category = list(categories)[0]
            auto_title = f"{category_title_map.get(category, '系统')}问题已修复"
        else:
            auto_title = "多个系统问题已修复"

        html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>发布系统公告</title>
                    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
                    <style>
                        body {{
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                            background: #f0f2f5;
                            padding: 2rem;
                        }}
                        .container {{
                            max-width: 700px;
                            margin: 0 auto;
                            background: white;
                            padding: 2rem;
                            border-radius: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        }}
                        h1 {{
                            color: #1a1f36;
                            margin-bottom: 1rem;
                            font-size: 1.5rem;
                        }}
                        .info {{
                            background: #f0f4ff;
                            border-left: 4px solid #4a6cf7;
                            padding: 1rem;
                            margin-bottom: 1.5rem;
                            border-radius: 4px;
                        }}
                        .form-group {{
                            margin-bottom: 1.5rem;
                        }}
                        label {{
                            display: block;
                            margin-bottom: 0.5rem;
                            font-weight: 600;
                            color: #374151;
                        }}
                        input, textarea {{
                            width: 100%;
                            padding: 0.75rem;
                            border: 2px solid #e5e7eb;
                            border-radius: 6px;
                            font-size: 1rem;
                            box-sizing: border-box;
                        }}
                        textarea {{
                            min-height: 150px;
                            resize: vertical;
                        }}
                        .auto-title-hint {{
                            font-size: 0.85rem;
                            color: #6b7280;
                            margin-top: 0.5rem;
                        }}
                        .btn-group {{
                            display: flex;
                            gap: 1rem;
                            margin-top: 2rem;
                        }}
                        .btn {{
                            padding: 0.75rem 2rem;
                            border: none;
                            border-radius: 6px;
                            font-size: 1rem;
                            font-weight: 600;
                            cursor: pointer;
                            flex: 1;
                        }}
                        .btn-primary {{
                            background: #4a6cf7;
                            color: white;
                        }}
                        .btn-primary:hover {{
                            background: #3b5de7;
                        }}
                        .btn-secondary {{
                            background: #e5e7eb;
                            color: #374151;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📢 发布系统公告</h1>
                        <div class="info">
                            <strong>已选中的错误日志ID：</strong> {ids}<br>
                            <strong>日志数量：</strong> {len(selected)} 条<br>
                            <strong>自动生成的标题：</strong> {auto_title}
                        </div>

                         <form method="post" action="/matting/submit-notice/">
                            <input type="hidden" name="csrfmiddlewaretoken" value="{request.META.get('CSRF_COOKIE', '')}">
                            <input type="hidden" name="log_ids" value="{ids}">

                            <div class="form-group">
                                <label>公告标题（可选，留空则自动生成）</label>
                                <input type="text" name="notice_title" placeholder="{auto_title}" value="{auto_title}">
                                <div class="auto-title-hint">
                                    💡 提示：系统已根据错误类型自动生成标题，您可以修改或直接使用
                                </div>
                            </div>
                            <div class="form-group">
                                <label>公告内容 <span style="color: red;">*</span></label>
                                <textarea name="notice_content" required placeholder="请输入公告内容，告知用户问题已解决..."></textarea>
                            </div>
                            <div class="btn-group">
                                <button type="submit" class="btn btn-primary">✅ 发布公告</button>
                                <a href="/admin/matting/mattingerrorlog/" class="btn btn-secondary" style="text-align: center; text-decoration: none;">❌ 取消</a>
                            </div>
                        </form>
                    </div>
                </body>
                </html>
                """
        return HttpResponse(html)

    publish_fix_notice.short_description = "📢 为已解决问题发布公告"

    def export_selected_errors(self, request, queryset):
        """导出选中的错误日志"""
        import csv
        from django.http import HttpResponse

        response = HttpResponse(content_type='text/csv; charset=utf-8-sig')
        response['Content-Disposition'] = 'attachment; filename="error_logs.csv"'

        writer = csv.writer(response)
        writer.writerow(['ID', '级别', '类别', '错误信息', '用户', '模型类型', '创建时间', '状态'])

        for obj in queryset:
            writer.writerow([
                obj.id,
                obj.get_error_level_display(),
                obj.get_error_category_display(),
                obj.error_message,
                obj.user.username if obj.user else '匿名',
                obj.model_type or '-',
                obj.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                '已解决' if obj.resolved else '未解决'
            ])

        self.message_user(request, f'成功导出 {queryset.count()} 条错误日志。')
        return response

    export_selected_errors.short_description = "导出选中的错误日志"

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related('user', 'resolved_by')

@admin.register(MattingResult)
class MattingResultAdmin(admin.ModelAdmin):
    list_display = (
        'user', 'original_image_preview', 'result_image_preview', 'model_type_badge', 'created_at', 'status')
    list_filter = ('model_type', 'status', 'created_at', 'is_deleted')
    search_fields = ('user__username', 'original_image', 'result_image')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

    # 添加字段显示
    fieldsets = (
        ('基本信息', {
            'fields': ('user', 'original_image', 'result_image', 'created_at'),
            'classes': ('collapse open',)
        }),
        ('模型信息', {
            'fields': ('model_type', 'model_info'),
            'description': '📊 记录抠图使用的模型类型和详细配置',
            'classes': ('collapse open',)
        }),
        ('背景设置', {
            'fields': ('background_type', 'background_color', 'has_transparent_version', 'transparent_image'),
            'description': '🎨 用户选择的背景类型和颜色',
            'classes': ('collapse open',)
        }),
        ('状态信息', {
            'fields': ('status', 'is_deleted', 'deleted_at', 'description'),
            'classes': ('collapse',)
        }),
    )

    # 只读字段
    readonly_fields = ('model_info_display', 'created_at', 'deleted_at')

    # 添加自定义动作
    actions = ['restore_selected_records']

    class Media:
        css = {
            'all': (
                'https://cdn.staticfile.org/font-awesome/6.4.0/css/all.min.css',
                'https://cdn.staticfile.org/simplelightbox/2.14.3/simple-lightbox.min.css'
            )
        }
        js = (
            'https://cdn.staticfile.org/simplelightbox/2.14.3/simple-lightbox.min.js',
        )

    def changelist_view(self, request, extra_context=None):
        """在列表页面初始化 SimpleLightbox"""
        from django.template.response import SimpleTemplateResponse

        response = super().changelist_view(request, extra_context)

        # 确保响应已经渲染
        if isinstance(response, SimpleTemplateResponse):
            response.render()

        # 现在可以安全地修改内容了
        if hasattr(response, 'content'):
            content = response.content.decode('utf-8')

            script = """
            <script>
            document.addEventListener('DOMContentLoaded', function() {
                if (typeof SimpleLightbox !== 'undefined') {
                    new SimpleLightbox('.image-preview-link', {
                        overlayOpacity: 0.8,
                        overlayBackground: '#fff',
                        closeBtnText: '<i class="fas fa-times"></i>',
                        navText: ['<i class="fas fa-chevron-left"></i>', '<i class="fas fa-chevron-right"></i>'],
                        showCounter: false,
                        animationSpeed: 200,
                        zoomable: true,
                        rotate: false,
                        scrollZoom: false,
                        doubleClickZoom: true,
                        swipeClose: true,
                        className: 'image-lightbox'
                    });
                }
            });
            </script>
            """

            content = content.replace('</body>', f'{script}</body>')
            response.content = content.encode('utf-8')

        return response

    def original_image_preview(self, obj):
        """原始图片预览"""
        if obj.original_image:
            return format_html(
                '<a href="{}" class="image-preview-link">'
                '<img src="{}" style="max-width:100px;max-height:100px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.15);cursor:pointer;transition:all 0.2s;" onmouseover="this.style.transform=\'scale(1.05)\',this.style.boxShadow=\'0 4px 12px rgba(0,0,0,0.25)\'" onmouseout="this.style.transform=\'scale(1)\',this.style.boxShadow=\'0 2px 6px rgba(0,0,0,0.15)\'" />'
                '</a>',
                obj.original_image.url,
                obj.original_image.url
            )
        return format_html('<span style="color:#999;">无图片</span>')

    original_image_preview.short_description = '原始图片'

    def result_image_preview(self, obj):
        """结果图片预览"""
        if obj.result_image:
            return format_html(
                '<a href="{}" class="image-preview-link">'
                '<img src="{}" style="max-width:100px;max-height:100px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.15);cursor:pointer;transition:all 0.2s;" onmouseover="this.style.transform=\'scale(1.05)\',this.style.boxShadow=\'0 4px 12px rgba(0,0,0,0.25)\'" onmouseout="this.style.transform=\'scale(1)\',this.style.boxShadow=\'0 2px 6px rgba(0,0,0,0.15)\'" />'
                '</a>',
                obj.result_image.url,
                obj.result_image.url
            )
        return format_html('<span style="color:#999;">无图片</span>')

    result_image_preview.short_description = '抠图结果'
    def model_type_badge(self, obj):
        """模型类型徽章显示"""
        model_config = {
            'default': {
                'icon': 'fa-balance-scale',
                'color': '#3b82f6',
                'bg_color': 'rgba(59, 130, 246, 0.1)',
                'name': '默认模型',
                'desc': '平衡性能与速度'
            },
            'high_quality': {
                'icon': 'fa-gem',
                'color': '#8b5cf6',
                'bg_color': 'rgba(139, 92, 246, 0.1)',
                'name': '高质量模型',
                'desc': '精度最高'
            },
            'lightweight': {
                'icon': 'fa-feather-alt',
                'color': '#06b6d4',
                'bg_color': 'rgba(6, 182, 212, 0.1)',
                'name': '轻量化模型',
                'desc': '速度最快'
            },
        }

        if obj.model_type in model_config:
            config = model_config[obj.model_type]
            return format_html(
                '<span style="display: inline-flex; align-items: center; gap: 6px; '
                'padding: 6px 14px; border-radius: 20px; '
                'background: {}; color: {}; font-weight: 600; font-size: 0.9rem; '
                'border: 1px solid {};">'
                '<i class="fas {}"></i>'
                '<div style="display: flex; flex-direction: column;">'
                '<span>{}</span>'
                '<span style="font-size: 0.75rem; opacity: 0.8; font-weight: normal;">{}</span>'
                '</div>'
                '</span>',
                config['bg_color'],
                config['color'],
                config['color'],
                config['icon'],
                config['name'],
                config['desc']
            )
        return obj.model_type

    model_type_badge.short_description = '使用模型'

    def background_type_badge(self, obj):
        """背景类型徽章显示"""
        if obj.background_type == 'transparent':
            return format_html(
                '<span style="display: inline-flex; align-items: center; gap: 6px; '
                'padding: 6px 14px; border-radius: 20px; '
                'background: rgba(6, 182, 212, 0.1); color: #06b6d4; font-weight: 600; font-size: 0.9rem; '
                'border: 1px solid #06b6d4;">'
                '<i class="fas fa-border-all"></i>'
                '<span>透明背景</span>'
                '</span>'
            )
        elif obj.background_type == 'solid_color':
            color = getattr(obj, 'background_color', '#FFFFFF')
            return format_html(
                '<span style="display: inline-flex; align-items: center; gap: 6px; '
                'padding: 6px 14px; border-radius: 20px; '
                'background: rgba(139, 92, 246, 0.1); color: #8b5cf6; font-weight: 600; font-size: 0.9rem; '
                'border: 1px solid #8b5cf6;">'
                '<i class="fas fa-palette"></i>'
                '<span>纯色背景</span>'
                '<span style="display: inline-block; width: 16px; height: 16px; border-radius: 4px; background: {}; border: 1px solid #ddd; margin-left: 6px;"></span>'
                '</span>',
                color
            )
        return obj.background_type

    background_type_badge.short_description = '背景类型'

    def model_info_display(self, obj):
        """模型详细信息显示"""
        if obj.model_info:
            import json
            try:
                info = json.loads(obj.model_info)

                html_parts = [
                    '<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); '
                    'padding: 20px; border-radius: 12px; border-left: 4px solid #4a6cf7;">'
                ]

                if 'name' in info:
                    html_parts.append(
                        f'<div style="margin-bottom: 15px;">'
                        f'<i class="fas fa-tag" style="color: #4a6cf7; margin-right: 8px;"></i>'
                        f'<strong style="color: #1a1f36;">模型名称：</strong>'
                        f'<span style="color: #555; margin-left: 8px;">{info["name"]}</span>'
                        f'</div>'
                    )

                if 'config_path' in info:
                    html_parts.append(
                        f'<div style="margin-bottom: 15px;">'
                        f'<i class="fas fa-file-code" style="color: #4a6cf7; margin-right: 8px;"></i>'
                        f'<strong style="color: #1a1f36;">配置文件：</strong>'
                        f'<div style="background: white; padding: 10px; border-radius: 6px; '
                        f'margin-top: 6px; font-family: monospace; font-size: 0.85rem; '
                        f'color: #666; word-break: break-all;">{info["config_path"]}</div>'
                        f'</div>'
                    )

                if 'checkpoint_path' in info:
                    html_parts.append(
                        f'<div style="margin-bottom: 15px;">'
                        f'<i class="fas fa-weight-hanging" style="color: #4a6cf7; margin-right: 8px;"></i>'
                        f'<strong style="color: #1a1f36;">权重文件：</strong>'
                        f'<div style="background: white; padding: 10px; border-radius: 6px; '
                        f'margin-top: 6px; font-family: monospace; font-size: 0.85rem; '
                        f'color: #666; word-break: break-all;">{info["checkpoint_path"]}</div>'
                        f'</div>'
                    )

                html_parts.append('</div>')
                return format_html(''.join(html_parts))
            except Exception as e:
                return format_html(
                    f'<div style="color: #dc3545; padding: 10px;">'
                    f'<i class="fas fa-exclamation-triangle"></i> '
                    f'解析模型信息失败：{str(e)}</div>'
                )
        return format_html('<span style="color: #999;">无详细信息</span>')

    model_info_display.short_description = '模型详细信息'

    def get_queryset(self, request):
        """默认显示所有记录（包括已删除的）"""
        return super().get_queryset(request)

    def restore_selected_records(self, request, queryset):
        """批量恢复选中的已删除记录"""
        restored_count = 0
        for record in queryset.filter(is_deleted=True):
            record.restore()
            restored_count += 1

        self.message_user(request, f'成功恢复了 {restored_count} 条记录。')

    restore_selected_records.short_description = "恢复选中的已删除记录"


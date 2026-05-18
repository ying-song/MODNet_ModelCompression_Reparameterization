from django.db import models
from django.conf import settings
from django.utils import timezone

class MattingResult(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    original_image = models.ImageField(upload_to='originals/')
    result_image = models.ImageField(upload_to='results/')
    created_at = models.DateTimeField(auto_now_add=True)
    # 添加软删除字段 - 确保是布尔类型
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(blank=True, null=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', '待处理'),
            ('processing', '处理中'),
            ('completed', '已完成'),
            ('failed', '失败'),
            ('deleted', '已删除')
        ],
        default='pending'
    )
    description = models.TextField(blank=True, null=True)
    # 新增：记录使用的模型信息
    model_type = models.CharField(
        max_length=50,
        choices=[
            ('default', '默认模型'),
            ('high_quality', '高质量模型'),
            ('lightweight', '轻量化模型')
        ],
        default='default'
    )
    model_info = models.TextField(blank=True, null=True, help_text="模型详细信息（如配置文件路径等）")

    # 背景相关字段
    background_type = models.CharField(
        max_length=20,
        choices=[
            ('transparent', '透明背景'),
            ('solid_color', '纯色背景'),
        ],
        default='transparent',
        verbose_name='背景类型'
    )
    background_color = models.CharField(
        max_length=7,
        default='#FFFFFF',
        help_text="纯色背景颜色，十六进制格式（如：#FFFFFF）",
        verbose_name='背景颜色'
    )
    has_transparent_version = models.BooleanField(default=False, verbose_name='是否有透明版本')
    transparent_image = models.ImageField(
        upload_to='results/transparent/',
        blank=True,
        null=True,
        verbose_name='透明背景图像'
    )

    def __str__(self):
        return f"{self.user.username} - {self.original_image.name}"

    def soft_delete(self):
        """软删除方法"""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.status = 'deleted'
        self.save()

    def restore(self):
        """恢复删除的记录"""
        self.is_deleted = False
        self.deleted_at = None
        if self.status == 'deleted':
            self.status = 'completed'
        self.save()

    @classmethod
    def get_normal_records(cls):
        """获取正常（未删除）的记录"""
        return cls.objects.filter(is_deleted=False)

    @classmethod
    def get_deleted_records(cls):
        """获取已删除的记录"""
        return cls.objects.filter(is_deleted=True)


class MattingErrorLog(models.Model):
    """抠图错误日志模型"""
    ERROR_LEVEL_CHOICES = [
        ('INFO', '信息'),
        ('WARNING', '警告'),
        ('ERROR', '错误'),
        ('CRITICAL', '严重错误'),
    ]

    ERROR_CATEGORY_CHOICES = [
        ('MODEL_LOAD', '模型加载失败'),
        ('IMAGE_PROCESS', '图像处理失败'),
        ('PREDICTION', '模型预测失败'),
        ('FILE_SAVE', '文件保存失败'),
        ('OTHER', '其他错误'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name='用户'
    )
    error_level = models.CharField(
        max_length=10,
        choices=ERROR_LEVEL_CHOICES,
        default='ERROR',
        verbose_name='错误级别'
    )
    error_category = models.CharField(
        max_length=20,
        choices=ERROR_CATEGORY_CHOICES,
        default='OTHER',
        verbose_name='错误类别'
    )
    error_message = models.TextField(verbose_name='错误信息')
    error_traceback = models.TextField(blank=True, null=True, verbose_name='错误堆栈')
    image_path = models.CharField(max_length=500, blank=True, null=True, verbose_name='图像路径')
    model_type = models.CharField(max_length=50, blank=True, null=True, verbose_name='模型类型')
    request_params = models.TextField(blank=True, null=True, help_text='请求参数 (JSON 格式)', verbose_name='请求参数')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    resolved = models.BooleanField(default=False, verbose_name='是否已解决')
    resolved_at = models.DateTimeField(blank=True, null=True, verbose_name='解决时间')
    resolved_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_logs',
        verbose_name='解决者'
    )
    notes = models.TextField(blank=True, null=True, help_text='管理员备注', verbose_name='备注')
    notify_users = models.BooleanField(default=False, verbose_name='是否通知所有用户')
    public_notice = models.TextField(blank=True, null=True, help_text='公开通知内容', verbose_name='公开通知')

    class Meta:
        ordering = ['-created_at']
        verbose_name = '抠图错误日志'
        verbose_name_plural = '抠图错误日志'

    def __str__(self):
        return f"[{self.error_level}] {self.error_category} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

    def mark_as_resolved(self, user=None, notes=None, notify_users=False, public_notice=None):
        """标记为已解决"""
        self.resolved = True
        self.resolved_at = timezone.now()
        self.resolved_by = user
        if notes:
            self.notes = notes
        self.notify_users = notify_users
        if public_notice:
            self.public_notice = public_notice
        self.save()

        if notify_users and public_notice:
            SystemNotice.objects.create(
                title=f"问题已解决：{self.error_category}",
                content=public_notice,
                notice_type='ERROR_RESOLVED',
                related_error_log=self,
                published_by=user,
                is_published=True,
                published_at=timezone.now()
            )

    @classmethod
    def get_unresolved(cls):
        """获取未解决的错误日志"""
        return cls.objects.filter(resolved=False)

    @classmethod
    def get_recent_errors(cls, days=7):
        """获取最近 N 天的错误日志"""
        from django.utils import timezone
        cutoff = timezone.now() - timezone.timedelta(days=days)
        return cls.objects.filter(created_at__gte=cutoff)


class UserMessage(models.Model):
    """用户与管理员之间的消息"""
    MESSAGE_TYPE_CHOICES = [
        ('USER_TO_ADMIN', '用户发给管理员'),
        ('ADMIN_TO_USER', '管理员回复用户'),
    ]

    STATUS_CHOICES = [
        ('UNREAD', '未读'),
        ('READ', '已读'),
        ('REPLIED', '已回复'),
    ]

    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='sent_messages',
        verbose_name='发送者',
        null=True,
        blank=True
    )

    recipient = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='received_messages',
        verbose_name='接收者',
        null=True,
        blank=True
    )
    message_type = models.CharField(
        max_length=20,
        choices=MESSAGE_TYPE_CHOICES,
        default='USER_TO_ADMIN',
        verbose_name='消息类型'
    )
    subject = models.CharField(max_length=200, verbose_name='主题')
    content = models.TextField(verbose_name='内容')
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='UNREAD',
        verbose_name='状态'
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    read_at = models.DateTimeField(blank=True, null=True, verbose_name='阅读时间')
    replied_message = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='replies',
        verbose_name='回复的消息'
    )
    related_error_log = models.ForeignKey(
        MattingErrorLog,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='messages',
        verbose_name='相关错误日志'
    )

    class Meta:
        ordering = ['-created_at']
        verbose_name = '用户消息'
        verbose_name_plural = '用户消息'

    def __str__(self):
        return f"{self.get_message_type_display()} - {self.subject}"

    def mark_as_read(self):
        """标记为已读"""
        self.status = 'READ'
        self.read_at = timezone.now()
        self.save()

    @classmethod
    def get_unread_count_for_user(cls, user):
        """获取用户的未读消息数"""
        return cls.objects.filter(recipient=user, status='UNREAD').count()

    @classmethod
    def get_user_messages(cls, user):
        """获取用户的所有消息（发送和接收）"""
        sent = cls.objects.filter(sender=user)
        received = cls.objects.filter(recipient=user)
        return (sent | received).distinct().order_by('-created_at')

    @classmethod
    def get_admin_messages(cls):
        """获取所有发给管理员的消息"""
        return cls.objects.filter(message_type='USER_TO_ADMIN')


class SystemNotice(models.Model):
    """系统公告（用于通知所有用户）"""
    NOTICE_TYPE_CHOICES = [
        ('GENERAL', '一般公告'),
        ('ERROR_RESOLVED', '问题已解决'),
        ('SYSTEM_MAINTENANCE', '系统维护'),
        ('FEATURE_UPDATE', '功能更新'),
    ]

    title = models.CharField(max_length=200, verbose_name='标题')
    content = models.TextField(verbose_name='内容')
    notice_type = models.CharField(
        max_length=30,
        choices=NOTICE_TYPE_CHOICES,
        default='GENERAL',
        verbose_name='公告类型'
    )
    is_published = models.BooleanField(default=False, verbose_name='是否发布')
    published_at = models.DateTimeField(blank=True, null=True, verbose_name='发布时间')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='创建时间')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    related_error_log = models.ForeignKey(
        MattingErrorLog,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='notices',
        verbose_name='相关错误日志'
    )
    published_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='published_notices',
        verbose_name='发布者'
    )

    class Meta:
        ordering = ['-published_at', '-created_at']
        verbose_name = '系统公告'
        verbose_name_plural = '系统公告'

    def __str__(self):
        return self.title

    def publish(self):
        """发布公告"""
        self.is_published = True
        self.published_at = timezone.now()
        self.save()

    @classmethod
    def get_published_notices(cls):
        """获取所有已发布的公告"""
        return cls.objects.filter(is_published=True)
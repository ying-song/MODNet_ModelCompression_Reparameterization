from django.contrib import admin
from django.utils.html import format_html
from .models import UploadedImage


@admin.register(UploadedImage)

class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ('user', 'image_preview', 'uploaded_at', 'status_badge')
    list_filter = ('status', 'uploaded_at')
    search_fields = ('user__username', 'image')
    date_hierarchy = 'uploaded_at'
    ordering = ('-uploaded_at',)

    # 字段显示配置
    fieldsets = (
        ('基本信息', {
            'fields': ('user', 'image', 'uploaded_at'),
            'classes': ('collapse open',)
        }),
        ('状态信息', {
            'fields': ('status',),
            'classes': ('collapse',)
        }),
    )

    # 只读字段
    readonly_fields = ('uploaded_at',)

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

    def image_preview(self, obj):
        """图片预览"""
        if obj.image:
            return format_html(
                '<a href="{}" class="image-preview-link">'
                '<img src="{}" style="max-width:100px;max-height:100px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.15);cursor:pointer;transition:all 0.2s;" onmouseover="this.style.transform=\'scale(1.05)\',this.style.boxShadow=\'0 4px 12px rgba(0,0,0,0.25)\'" onmouseout="this.style.transform=\'scale(1)\',this.style.boxShadow=\'0 2px 6px rgba(0,0,0,0.15)\'" />'
                '</a>',
                obj.image.url,
                obj.image.url
            )
        return format_html('<span style="color:#999;">无图片</span>')

    image_preview.short_description = '上传图片'

    def status_badge(self, obj):
        """状态徽章显示"""
        status_config = {
            'pending': {
                'icon': 'fa-clock',
                'color': '#f59e0b',
                'bg_color': 'rgba(245, 158, 11, 0.1)',
                'name': '待处理'
            },
            'processing': {
                'icon': 'fa-cog fa-spin',
                'color': '#3b82f6',
                'bg_color': 'rgba(59, 130, 246, 0.1)',
                'name': '处理中'
            },
            'completed': {
                'icon': 'fa-check-circle',
                'color': '#10b981',
                'bg_color': 'rgba(16, 185, 129, 0.1)',
                'name': '已完成'
            },
            'failed': {
                'icon': 'fa-times-circle',
                'color': '#ef4444',
                'bg_color': 'rgba(239, 68, 68, 0.1)',
                'name': '失败'
            },
        }

        if obj.status in status_config:
            config = status_config[obj.status]
            return format_html(
                '<span style="display: inline-flex; align-items: center; gap: 6px; '
                'padding: 6px 14px; border-radius: 20px; '
                'background: {}; color: {}; font-weight: 600; font-size: 0.9rem; '
                'border: 1px solid {};">'
                '<i class="fas {}"></i>'
                '<span>{}</span>'
                '</span>',
                config['bg_color'],
                config['color'],
                config['color'],
                config['icon'],
                config['name']
            )
        return obj.status

    status_badge.short_description = '状态'


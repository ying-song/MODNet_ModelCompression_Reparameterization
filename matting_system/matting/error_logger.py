import json
import traceback
from functools import wraps
from django.utils import timezone
from .models import MattingErrorLog


def log_matting_error(error_category='OTHER'):
    """
    装饰器：自动记录抠图过程中的错误

    Usage:
        @log_matting_error(error_category='MODEL_LOAD')
        def load_model(model_type):
            ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 提取请求信息
                request = None
                for arg in args:
                    if hasattr(arg, 'user') and hasattr(arg, 'method'):
                        request = arg
                        break

                # 获取用户信息
                user = None
                if request and hasattr(request, 'user'):
                    user = request.user if request.user.is_authenticated else None

                # 获取模型类型
                model_type = kwargs.get('model_type', args[-1] if len(args) > 0 else None)

                # 获取图像路径
                image_path = kwargs.get('image_path', args[0] if len(args) > 0 else None)
                if hasattr(image_path, 'path'):
                    image_path = image_path.path

                # 获取请求参数
                request_params = None
                if request and hasattr(request, 'GET'):
                    request_params = json.dumps(dict(request.GET), ensure_ascii=False)
                elif request and hasattr(request, 'POST'):
                    request_params = json.dumps(dict(request.POST), ensure_ascii=False)

                # 创建错误日志
                error_log = MattingErrorLog.objects.create(
                    user=user,
                    error_level='ERROR',
                    error_category=error_category,
                    error_message=str(e),
                    error_traceback=traceback.format_exc(),
                    image_path=str(image_path) if image_path else None,
                    model_type=str(model_type) if model_type else None,
                    request_params=request_params
                )

                # 重新抛出异常
                raise

        return wrapper

    return decorator


def create_error_log(error, category='OTHER', user=None, image_path=None,
                     model_type=None, request_params=None, error_level='ERROR'):
    """
    手动创建错误日志

    Usage:
        try:
            # some code
        except Exception as e:
            create_error_log(e, category='IMAGE_PROCESS', user=request.user)
    """
    import traceback

    # 转换 image_path
    if hasattr(image_path, 'path'):
        image_path = image_path.path

    # 转换 request_params 为 JSON
    if request_params and isinstance(request_params, dict):
        request_params = json.dumps(request_params, ensure_ascii=False)

    error_log = MattingErrorLog.objects.create(
        user=user,
        error_level=error_level,
        error_category=category,
        error_message=str(error),
        error_traceback=traceback.format_exc(),
        image_path=str(image_path) if image_path else None,
        model_type=str(model_type) if model_type else None,
        request_params=request_params
    )

    return error_log

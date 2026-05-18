from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import numpy as np
from PIL import Image
import torch, cv2, os, sys, traceback, json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from .models import MattingResult, MattingErrorLog
from .error_logger import create_error_log, log_matting_error

try:
    from uploads.models import UploadedImage
except ImportError:
    try:
        from ..uploads.models import UploadedImage
    except ImportError:
        raise ImportError("无法导入 UploadedImage 模型")

sys.path.append(os.path.join(BASE_DIR, '..'))
try:
    from infer2 import predit_matte
except ImportError:
    import importlib
    import sys

    sys.path.append(BASE_DIR)
    predit_matte = importlib.import_module('infer2').predit_matte

# 导入模型配置
from .model_registry import ModelConfig

# 全局变量存储模型缓存
model_cache = {}


def load_model(model_type='default'):
    """加载指定类型的 MODNet 模型"""
    global model_cache

    if model_type not in model_cache:
        try:
            model, config = ModelConfig.load_model(model_type)
            model_cache[model_type] = (model, config)
        except Exception as e:
            print(f"模型加载失败：{e}")
            return None, None

    return model_cache[model_type]

def generate_matting(request, image_id):
    try:
        uploaded_image = UploadedImage.objects.get(id=image_id)

        # 获取用户选择的模型类型（从 GET 参数或 POST 数据）
        model_type = request.GET.get('model_type', request.POST.get('model_type', 'default'))

        # 读取原始图像
        original_path = uploaded_image.image.path

        # 调用你的 MODNet 模型进行抠图
        result_path = generate_matting_with_model(original_path, model_type, request.user)

        # 创建抠图结果记录
        matting_result = MattingResult(
            user=request.user if request.user.is_authenticated else None,
            original_image=uploaded_image,
            result_image=result_path.replace('media/', ''),
            preview_image=result_path.replace('media/', ''),
            model_type=model_type,
            description=f'使用{ModelConfig.get_model_config(model_type)["name"]}生成'
        )
        matting_result.save()

        messages.success(request, f'人像抠图生成成功！使用模型：{ModelConfig.get_model_config(model_type)["name"]}')
        return redirect('matting:result', matting_result.id)

    except UploadedImage.DoesNotExist:
        error_msg = '图片不存在'
        create_error_log(
            Exception(error_msg),
            category='IMAGE_PROCESS',
            user=request.user if request.user.is_authenticated else None,
            model_type=request.GET.get('model_type', 'default'),
            request_params={'image_id': image_id}
        )
        messages.error(request, error_msg)
        return redirect('uploads:upload')

    except Exception as e:
        error_msg = f'人像抠图生成失败：{str(e)}'
        create_error_log(
            e,
            category='PREDICTION',
            user=request.user if request.user.is_authenticated else None,
            model_type=request.GET.get('model_type', 'default'),
            image_path=uploaded_image.image.path if 'uploaded_image' in locals() else None,
            request_params={
                'image_id': image_id,
                'model_type': request.GET.get('model_type', 'default')
            }
        )
        messages.error(request, error_msg)
        return redirect('uploads:upload')


def generate_matting_with_model(image_path, model_type='default', user=None):
    """使用 MODNet 模型进行人像抠图，返回抠图结果和 alpha matte"""
    try:
        # 加载模型
        modnet, config = load_model(model_type)
        if modnet is None:
            raise Exception("模型加载失败")

        # 执行抠图
        img = Image.open(image_path)
        matte = predit_matte(modnet, img)

        # 生成结果图像
        h, w = matte.shape[:2]

        prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        prd_img_cv = cv2.cvtColor(np.asarray(prd_img), cv2.COLOR_RGB2BGR)
        res = cv2.add(img_cv, 255 - prd_img_cv)

        # 保存结果
        result_path = image_path.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
        save_success = cv2.imwrite(result_path, res)

        if not save_success:
            raise Exception(f"保存图像失败：{result_path}")

        return result_path, matte

    except FileNotFoundError as e:
        create_error_log(
            e,
            category='FILE_SAVE',
            user=user,
            image_path=image_path,
            model_type=model_type,
            error_level='ERROR'
        )
        raise

    except Exception as e:
        create_error_log(
            e,
            category='PREDICTION',
            user=user,
            image_path=image_path,
            model_type=model_type,
            error_level='ERROR'
        )
        raise


def apply_background(image_path, matte, background_type='transparent', background_color='#FFFFFF'):
    """应用背景到抠图结果"""
    try:
        # 读取原始图像
        img = Image.open(image_path)
        img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]

        # 处理 alpha matte
        if matte.shape[:2] != (h, w):
            matte = cv2.resize(matte, (w, h))

        # 创建 alpha 通道
        alpha = (matte * 255).astype('uint8')

        if background_type == 'transparent':
            # 创建透明背景图像（PNG 格式）
            b, g, r = cv2.split(img_cv)
            rgba = [b, g, r, alpha]
            dst = cv2.merge(rgba, 4)

            # 保存为 PNG
            result_path = image_path.replace('.jpg', '_transparent.png').replace('.png', '_transparent.png')
            cv2.imwrite(result_path, dst)


        else:  # solid_color
            # 解析颜色值
            hex_color = background_color.lstrip('#')
            # RGB 转 BGR（OpenCV 使用 BGR 格式）
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            bg_color = (b, g, r)  # BGR 格式

            # 创建纯色背景
            background = np.zeros((h, w, 3), dtype=np.uint8)
            background[:] = bg_color

            # 混合前景和背景
            alpha_3ch = alpha.astype(float) / 255.0
            alpha_3ch = cv2.merge([alpha_3ch, alpha_3ch, alpha_3ch])

            foreground = img_cv.astype(float) * alpha_3ch
            background_layer = background.astype(float) * (1 - alpha_3ch)
            dst = (foreground + background_layer).astype('uint8')

            # 保存到 uploads 目录，确保 URL 可以访问
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir, exist_ok=True)

            # 生成唯一文件名
            import uuid
            filename = f'solid_{uuid.uuid4().hex[:8]}.jpg'
            result_path = os.path.join(upload_dir, filename)
            cv2.imwrite(result_path, dst)

        return result_path

    except Exception as e:
        create_error_log(
            e,
            category='IMAGE_PROCESS',
            user=None,
            image_path=image_path,
            error_level='ERROR'
        )
        raise


@require_POST
def change_background(request, result_id):
    """更改抠图结果的背景"""
    try:
        matting_result = MattingResult.objects.get(id=result_id)

        background_type = request.POST.get('background_type', 'transparent')
        background_color = request.POST.get('background_color', '#FFFFFF')

        # 获取原始图像路径
        original_path = matting_result.original_image.path

        # 重新加载 matte（如果之前没有保存，需要重新计算）
        modnet, config = load_model(matting_result.model_type)
        img = Image.open(original_path)
        matte = predit_matte(modnet, img)

        # 应用新背景
        new_result_path = apply_background(
            original_path,
            matte,
            background_type,
            background_color
        )

        # 更新记录
        if background_type == 'transparent':
            matting_result.transparent_image = new_result_path.replace('media/', '')
            matting_result.has_transparent_version = True
        else:
            matting_result.result_image = new_result_path.replace('media/', '')

        matting_result.background_type = background_type
        matting_result.background_color = background_color
        matting_result.save()

        # 构建正确的媒体 URL
        import os
        relative_path = os.path.join('uploads', os.path.basename(new_result_path))
        media_url = f'{settings.MEDIA_URL}uploads/{os.path.basename(new_result_path)}'

        # 验证文件是否存在
        if not os.path.exists(new_result_path):
            raise FileNotFoundError(f"保存的文件不存在：{new_result_path}")

        return JsonResponse({
            'success': True,
            'message': '背景更换成功',
            'image_url': media_url,
            'file_path': new_result_path  # 调试用，可以移除
        })

    except MattingResult.DoesNotExist:
        return JsonResponse({
            'success': False,
            'message': '记录不存在'
        }, status=404)
    except FileNotFoundError as e:
        return JsonResponse({
            'success': False,
            'message': f'文件保存失败：{str(e)}'
        }, status=500)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'更换背景失败：{str(e)}'
        }, status=500)


def result(request, result_id):
    """显示抠图结果"""
    try:
        matting_result = MattingResult.objects.get(id=result_id)
        context = {
            'matting_result': matting_result
        }
        return render(request, 'matting_system/templates/matting/result.html', context)
    except MattingResult.DoesNotExist:
        messages.error(request, '找不到指定的抠图结果')
        return redirect('uploads:upload')


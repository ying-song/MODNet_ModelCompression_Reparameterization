from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.http import JsonResponse

from .models import UploadedImage
try:
    from matting.models import MattingResult
    from matting.error_logger import create_error_log
except ImportError:
    # 如果上面的导入失败，尝试使用绝对路径导入
    from matting_system.matting.models import MattingResult
    from matting_system.matting.error_logger import create_error_log

from PIL import Image
import torch, os, json, sys
import numpy as np
import cv2 as cv
from django.conf import settings


# 导入 infer2.py 中的函数和模型定义
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from infer2 import predit_matte, MODNet_auto

# 导入模型配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from matting.model_registry import ModelConfig
except ImportError:
    # 如果上面的导入失败，尝试使用绝对路径导入
    from matting_system.matting.model_registry import ModelConfig


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # 验证图像格式和大小
        image = request.FILES['image']
        allowed_formats = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
        file_extension = image.name.split('.')[-1].lower()

        if file_extension not in allowed_formats:
            messages.error(request, '不支持的图片格式。请上传 JPG、PNG、GIF 或 BMP 格式的图片。')
            return redirect('uploads:upload')

        # 验证图片大小（例如：不超过 5MB）
        if image.size > 5 * 1024 * 1024:
            messages.error(request, '图片大小不能超过 5MB。')
            return redirect('uploads:upload')

        # 获取用户选择的模型类型
        model_type = request.POST.get('model_type', 'default')

        # 保存图片
        uploaded_image = UploadedImage(
            user=request.user if request.user.is_authenticated else None,
            image=image
        )
        uploaded_image.save()

        # 调用抠图模型
        try:
            # 加载选定类型的模型
            modnet, model_config = ModelConfig.load_model(model_type)

            img = Image.open(image)
            matte = predit_matte(modnet, img)

            # 保存结果图
            prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
            prd_img = np.asarray(prd_img)

            img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
            mask = cv.merge([prd_img, prd_img, prd_img])
            res = cv.add(img, 255 - mask)

            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir, exist_ok=True)

            result_filename = f'result_{uploaded_image.id}.{file_extension}'
            result_path = os.path.join(upload_dir, result_filename)

            save_success = cv.imwrite(result_path, res)
            if save_success:
                print(f"Successfully saved result image: {result_path}")
            else:
                raise Exception(f"保存图像失败：{result_path}")

            # 同时生成透明背景版本
            try:
                h, w = matte.shape[:2]
                alpha = (matte * 255).astype('uint8')

                b, g, r = cv.split(img)
                rgba = [b, g, r, alpha]
                transparent_img = cv.merge(rgba, 4)

                transparent_filename = f'transparent_{uploaded_image.id}.png'
                transparent_path = os.path.join(upload_dir, transparent_filename)

                cv.imwrite(transparent_path, transparent_img)

                # 创建或更新 MattingResult 记录
                matting_result, created = MattingResult.objects.get_or_create(
                    user=uploaded_image.user if uploaded_image.user else None,
                    original_image=uploaded_image.image,
                    defaults={
                        'status': 'completed',
                        'description': '自动创建的抠图结果',
                        'model_type': model_type,
                        'background_type': 'transparent',
                        'has_transparent_version': True,
                        'transparent_image': f'uploads/{transparent_filename}',
                        'result_image': f'uploads/{result_filename}',
                        'model_info': json.dumps({
                            'name': model_config['name'],
                            'config_path': model_config['prune_info_path'],
                            'checkpoint_path': model_config['ckp_pth']
                        })
                    }
                )

                # 更新结果图像字段
                matting_result.result_image = f'uploads/{result_filename}'
                matting_result.status = 'completed'
                matting_result.model_type = model_type
                matting_result.save()

            except Exception as e:
                print(f"生成透明背景失败：{e}")
                # 即使透明背景生成失败，也要保证基本流程继续

            messages.success(request, f'图片上传并抠图成功！使用模型：{model_config["name"]}')

        except FileNotFoundError as e:
            error_msg = f'文件操作失败：{str(e)}'
            create_error_log(
                e,
                category='FILE_SAVE',
                user=request.user if request.user.is_authenticated else None,
                image_path=image.path if hasattr(image, 'path') else image.name,
                model_type=model_type,
                request_params={'model_type': model_type}
            )
            messages.error(request, error_msg)
            uploaded_image.status = 'failed'
            uploaded_image.save()

        except Exception as e:
            error_msg = f'抠图过程中发生错误：{str(e)}'
            create_error_log(
                e,
                category='PREDICTION',
                user=request.user if request.user.is_authenticated else None,
                image_path=image.path if hasattr(image, 'path') else image.name,
                model_type=model_type,
                request_params={'model_type': model_type}
            )
            messages.error(request, error_msg)
            uploaded_image.status = 'failed'
            uploaded_image.save()

        return redirect('uploads:success', uploaded_image.id)

    return render(request, 'uploads/upload.html')


def upload_success(request, image_id):
    try:
        # 获取原始图像记录
        uploaded_image = UploadedImage.objects.get(id=image_id)

        # 获取对应的抠图结果记录
        matting_result = MattingResult.objects.filter(original_image=uploaded_image.image).first()

        # 准备上下文数据
        context = {
            'image': uploaded_image,
            'matting_result': matting_result
        }

        return render(request, 'uploads/success.html', context)
    except UploadedImage.DoesNotExist:
        messages.error(request, '图片不存在。')
        return redirect('uploads:upload')


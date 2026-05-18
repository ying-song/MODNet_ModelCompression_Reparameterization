from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('matting', '0006_mattingerrorlog'),
    ]

    operations = [
        migrations.AddField(
            model_name='mattingresult',
            name='background_type',
            field=models.CharField(
                choices=[('transparent', '透明背景'), ('solid_color', '纯色背景')],
                default='transparent',
                max_length=20,
                verbose_name='背景类型'
            ),
        ),
        migrations.AddField(
            model_name='mattingresult',
            name='background_color',
            field=models.CharField(
                default='#FFFFFF',
                help_text='纯色背景颜色，十六进制格式（如：#FFFFFF）',
                max_length=7,
                verbose_name='背景颜色'
            ),
        ),
        migrations.AddField(
            model_name='mattingresult',
            name='has_transparent_version',
            field=models.BooleanField(default=False, verbose_name='是否有透明版本'),
        ),
        migrations.AddField(
            model_name='mattingresult',
            name='transparent_image',
            field=models.ImageField(
                blank=True,
                null=True,
                upload_to='results/transparent/',
                verbose_name='透明背景图像'
            ),
        ),
    ]

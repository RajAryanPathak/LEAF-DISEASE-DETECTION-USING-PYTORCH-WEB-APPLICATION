# Generated by Django 3.0.7 on 2020-10-28 09:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('leafdis', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='forum',
            name='post',
            field=models.TextField(default='', max_length=1000, null=True),
        ),
    ]

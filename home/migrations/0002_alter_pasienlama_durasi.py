# Generated by Django 4.2.4 on 2023-11-05 17:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pasienlama',
            name='DURASI',
            field=models.TimeField(),
        ),
    ]
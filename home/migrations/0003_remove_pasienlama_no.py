# Generated by Django 4.2.4 on 2023-11-06 17:14

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0002_alter_pasienlama_durasi'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='pasienlama',
            name='NO',
        ),
    ]

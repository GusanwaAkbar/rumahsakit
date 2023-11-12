# Generated by Django 4.2.4 on 2023-11-05 10:54

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PasienLama',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('NO', models.IntegerField()),
                ('NORM', models.IntegerField()),
                ('TGL_PELAYANAN', models.CharField(max_length=255)),
                ('NAMA_PASIEN', models.CharField(max_length=255)),
                ('POLI_TUJUAN', models.CharField(max_length=255)),
                ('JAM_DAFTAR', models.TimeField()),
                ('JAM_MAP_TERSEDIA', models.TimeField()),
                ('DURASI', models.DurationField()),
            ],
        ),
    ]
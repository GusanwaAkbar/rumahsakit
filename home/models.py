from django.db import models


class PasienLama(models.Model):
    NORM = models.IntegerField()
    TGL_PELAYANAN = models.CharField(max_length=255)
    NAMA_PASIEN = models.CharField(max_length=255)
    POLI_TUJUAN = models.CharField(max_length=255)
    JAM_DAFTAR = models.TimeField()
    JAM_MAP_TERSEDIA = models.TimeField()
    DURASI = models.TimeField()


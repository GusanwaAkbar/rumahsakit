import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from home.models import PasienLama

class Command(BaseCommand):
    help = 'Import data from CSV into the PasienLama model'

    def handle(self, *args, **options):
        csv_file_path = '/home/gusanwa/AA_Programming/rumahsakit/rs/notebook/pasienlama2.csv'  # Update this with the actual path to your CSV file

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # Assuming your CSV has columns NORM, TGL_PELAYANAN, NAMA_PASIEN, POLI_TUJUAN, JAM_DAFTAR, JAM_MAP_TERSEDIA, DURASI
                PasienLama.objects.create(
                    NORM=int(row['NORM']),
                    TGL_PELAYANAN=row['TGL PELAYANAN'],
                    NAMA_PASIEN=row['NAMA PASIEN'],
                    POLI_TUJUAN=row['POLI TUJUAN'],
                    JAM_DAFTAR=datetime.strptime(row['JAM DAFTAR'], '%H:%M:%S').time(),
                    JAM_MAP_TERSEDIA=datetime.strptime(row['JAM MAP TERSEDIA'], '%H:%M:%S').time(),
                    DURASI=datetime.strptime(row['DURASI'], '%H:%M:%S').time(),
                )

        self.stdout.write(self.style.SUCCESS('Data imported successfully'))

TARGET_NAME = 'NObeyesdad'

CONTINUOUS_COLS = ['Age', 'Height', 'Weight']
CATEGORICAL_COLS = [
    'Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
    'family_history_with_overweight', 'CAEC', 'MTRANS'
]
ORDINAL_COLS = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
ALL_CATEGORICAL_COLS = CATEGORICAL_COLS + ORDINAL_COLS

# Hierarchy obesitas dari terburuk ke terbaik
OBESITY_REDUCTION_HIERARCHY = [
    'Obesity_Type_III',
    'Obesity_Type_II', 
    'Obesity_Type_I',
    'Overweight',
    'Normal_Weight'
]

# Hirarki untuk menaikkan berat badan
OBESITY_GAIN_HIERARCHY = [
    'Insufficient_Weight',
    'Normal_Weight'
]

# Mapping untuk encoding
GENDER_MAP = {'Female': 0, 'Male': 1}
FAMILY_HISTORY_MAP = {'no': 0, 'yes': 1}
FAVC_MAP = {'no': 0, 'yes': 1}
SCC_MAP = {'no': 0, 'yes': 1}
SMOKE_MAP = {'no': 0, 'yes': 1}
CAEC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
CALC_MAP = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
MTRANS_MAP = {
    'Walking': 0,
    'Public_Transportation': 1,
    'Bike': 2,
    'Motorbike': 3,
    'Automobile': 4
}

# Mapping untuk decode
DECODE_MAPS = {
    'Gender': {0: 'Female', 1: 'Male'},
    'family_history_with_overweight': {0: 'no', 1: 'yes'},
    'FAVC': {0: 'no', 1: 'yes'},
    'SCC': {0: 'no', 1: 'yes'},
    'SMOKE': {0: 'no', 1: 'yes'},
    'CAEC': {0: 'no', 1: 'Sometimes', 2: 'Frequently', 3: 'Always'},
    'CALC': {0: 'no', 1: 'Sometimes', 2: 'Frequently', 3: 'Always'},
    'MTRANS': {0: 'Walking', 1: 'Public_Transportation', 2: 'Bike', 3: 'Motorbike', 4: 'Automobile'}
}

# ==============================================================================
# TAMBAHAN BARU: ACTION VERBS
# Digunakan untuk membuat kalimat saran yang manusiawi di dice_helpers.py
# ==============================================================================
ACTION_VERBS = {
    'Weight': "Targetkan penurunan berat badan",
    'FCVC': "Tingkatkan konsumsi sayuran",
    'FAVC': "Kurangi konsumsi makanan cepat saji (junk food)",
    'NCP': "Atur jadwal makan utama",
    'CAEC': "Kontrol kebiasaan mengemil (ngemil)",
    'CH2O': "Perbanyak minum air putih",
    'FAF': "Tingkatkan frekuensi aktivitas fisik/olahraga",
    'TUE': "Kurangi waktu penggunaan gawai (HP/PC)",
    'CALC': "Kurangi konsumsi alkohol",
    'SCC': "Mulai memantau asupan kalori harian",
    'SMOKE': "Berhenti merokok",
    'MTRANS': "Ubah moda transportasi (perbanyak jalan kaki jika mungkin)"
}
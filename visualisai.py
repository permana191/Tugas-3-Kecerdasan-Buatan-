import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv('dataset.csv')

# =========================
# 2. Preprocessing
# =========================
# Filter hanya Jawa Barat (opsional, tapi dataset kamu memang Jabar)
data = data[data['nama_provinsi'] == 'JAWA BARAT']

# Agregasi: rata-rata konsumsi per tahun
data_grouped = data.groupby('tahun')['total'].mean().reset_index()

# =========================
# 3. Menentukan Variabel
# =========================
X = data_grouped[['tahun']]   # HARUS 2D
y = data_grouped['total']

# =========================
# 4. Membuat Model Regresi
# =========================
model = LinearRegression()
model.fit(X, y)

# =========================
# 5. Prediksi
# =========================
y_pred = model.predict(X)

# =========================
# 6. Visualisasi
# =========================
plt.figure(figsize=(10, 6))

# Scatter data asli
plt.scatter(X, y, color='blue', label='Data Aktual')

# Garis regresi
plt.plot(X, y_pred, color='red', linewidth=2, label='Garis Regresi Linear')

# Judul & Label
plt.title('Tren Konsumsi Minyak & Lemak di Jawa Barat (2019-2025)')
plt.xlabel('Tahun')
plt.ylabel('Konsumsi (KG/KAP/TAHUN)')

# Tambahan
plt.legend()
plt.grid(True)

# Tampilkan grafik
plt.show()

# =========================
# 7. Output Model (opsional buat laporan)
# =========================
print("Koefisien (b):", model.coef_[0])
print("Intercept (a):", model.intercept_)
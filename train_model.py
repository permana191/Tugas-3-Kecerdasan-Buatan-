import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

print("Memulai proses training model (Versi Bebas Outlier 2021-2022)...")

# 1. Membaca Dataset
try:
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("Error: File 'dataset.csv' tidak ditemukan.")
    exit()

# 2. Preprocessing: Menghitung rata-rata konsumsi per tahun
data_tahunan = df.groupby('tahun')['total'].mean().reset_index()

# 3. DATA CLEANING: Menghapus Outlier (Masa Krisis Minyak Goreng 2021-2022)
# Kita mengecualikan tahun 2021 dan 2022 karena terjadi anomali pasar
data_bersih = data_tahunan[~data_tahunan['tahun'].isin([2021, 2022])]

# 4. Menentukan Variabel X dan Y (menggunakan data yang sudah dibersihkan)
X = data_bersih[['tahun']]
y = data_bersih['total']

# 5. Melatih Model Regresi Linear
model = LinearRegression()
model.fit(X, y)

# 6. Evaluasi Model
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nHasil Evaluasi Model")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared (R2): {r2:.2f}")

# 7. Menyimpan Model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n!")
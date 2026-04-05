from flask import Flask, render_template, request
import pickle
import pandas as pd
import io
import base64

# Konfigurasi Matplotlib untuk server web agar tidak crash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model regresi
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: File 'model.pkl' tidak ditemukan!")

# Load dataset untuk ditampilkan di tabel dan grafik
df = pd.read_csv('dataset.csv')
data_tahunan = df.groupby('tahun')['total'].mean().reset_index()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediksi = None
    tahun_input = None
    
    # 1. Membuat Grafik untuk ditampilkan di Web
    plt.figure(figsize=(8, 4))
    
    # Menyiapkan data Sumbu X dan Y
    x_axis = data_tahunan['tahun']
    y_aktual = data_tahunan['total']
    
    # Prediksi untuk garis merah (menggunakan DataFrame untuk menghindari warning)
    # Gunakan flatten() agar output 2D menjadi 1D sesuai permintaan Matplotlib
    y_prediksi = model.predict(data_tahunan[['tahun']]).flatten()
    
    plt.scatter(x_axis, y_aktual, color='blue', label='Data Aktual')
    plt.plot(x_axis, y_prediksi, color='red', label='Garis Regresi')
    plt.title('Tren Konsumsi Minyak & Lemak (Jabar)')
    plt.xlabel('Tahun')
    plt.ylabel('Konsumsi (KG/KAP/TAHUN)')
    plt.legend()
    plt.grid(True)
    
    # Mengubah grafik menjadi format gambar (base64) agar bisa dibaca HTML
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close() # Menutup grafik dari memori

    # 2. Menangani Input User dari Form
    if request.method == 'POST':
        tahun_input = int(request.form['tahun'])
        
        # Membungkus input dalam format DataFrame agar formatnya sama saat training model
        input_df = pd.DataFrame({'tahun': [tahun_input]})
        
        # Prediksi dan ambil nilai pertamanya, lalu bulatkan
        hasil_prediksi = model.predict(input_df).flatten()
        prediksi = round(hasil_prediksi[0], 2)
        
    return render_template('index.html', 
                           plot_url=plot_url, 
                           data=data_tahunan.to_dict(orient='records'),
                           prediksi=prediksi, 
                           tahun=tahun_input)

if __name__ == "__main__":
    app.run(debug=True)
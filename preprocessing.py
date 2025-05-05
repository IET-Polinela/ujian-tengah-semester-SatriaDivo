import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Membaca dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv')

# 1. Data Cleaning (Memperbaiki data yang tidak konsisten)
# Menghapus baris dengan data yang tidak valid
data = data.dropna(subset=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type'])

# 2. Handling Missing Values (Menangani nilai yang hilang)
# Ganti nilai NaN dengan median pada kolom numerik
data['bmi'].fillna(data['bmi'].median(), inplace=True)
data['avg_glucose_level'].fillna(data['avg_glucose_level'].median(), inplace=True)

# 3. Outliers (Mengatasi nilai ekstrem)
# Menggunakan IQR untuk mengidentifikasi dan menghapus outlier
Q1 = data[['age', 'bmi', 'avg_glucose_level']].quantile(0.25)
Q3 = data[['age', 'bmi', 'avg_glucose_level']].quantile(0.75)
IQR = Q3 - Q1

# Hapus baris dengan nilai outlier
data = data[~((data[['age', 'bmi', 'avg_glucose_level']] < (Q1 - 1.5 * IQR)) | 
              (data[['age', 'bmi', 'avg_glucose_level']] > (Q3 + 1.5 * IQR))).any(axis=1)]

# 4. Feature Scaling (Menstandarkan data)
# Standardization menggunakan StandardScaler
scaler = StandardScaler()
data[['age', 'avg_glucose_level', 'bmi']] = scaler.fit_transform(data[['age', 'avg_glucose_level', 'bmi']])

# 5. Dimensionality Reduction (PCA)
pca = PCA(n_components=2)  # Mengurangi dimensi menjadi 2 komponen
pca_components = pca.fit_transform(data[['age', 'avg_glucose_level', 'bmi']])
data['pca_1'] = pca_components[:, 0]
data['pca_2'] = pca_components[:, 1]

# 6. t-SNE untuk visualisasi 2D
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(data[['age', 'avg_glucose_level', 'bmi']])
data['tsne_1'] = tsne_components[:, 0]
data['tsne_2'] = tsne_components[:, 1]

# 7. Feature Engineering (Menambah fitur baru)
# Menambahkan fitur baru: BMI kategori (underweight, normal, overweight, obese)
def bmi_category(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif 18.5 <= bmi < 24.9:
        return 'normal'
    elif 25 <= bmi < 29.9:
        return 'overweight'
    else:
        return 'obese'

data['bmi_category'] = data['bmi'].apply(bmi_category)

# 8. Feature Selection (Memilih fitur terbaik)
X = data.drop(['stroke'], axis=1)  # Fitur
y = data['stroke']  # Target

# 9. Membagi data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 10. Menyimpan data yang sudah diproses ke CSV
data.to_csv('preprocessed_data_standard_scaling.csv', index=False)

# Cek hasil dan pastikan data sudah disimpan
print("Data telah disimpan dalam 'preprocessed_data_standard_scaling.csv'.")

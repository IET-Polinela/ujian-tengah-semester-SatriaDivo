import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('preprocessed_data_standard_scaling.csv')

# Drop 'id' karena tidak relevan
df.drop(columns=['id'], inplace=True)

# Imputasi nilai NaN pada BMI
imputer = SimpleImputer(strategy='mean')
df['bmi'] = imputer.fit_transform(df[['bmi']])

# One-hot encode 'bmi_category' dan fitur kategorikal lainnya
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'bmi_category']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # sparse=False for KMeans compatibility

encoded_data = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

# Gabungkan fitur encoded dengan fitur numerik
processed_df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# Coba beberapa nilai k untuk KMeans (k=2 hingga k=5)
sse = []
silhouette_scores = []
K = range(2, 6)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(processed_df)
    sse.append(kmeans.inertia_)
    score = silhouette_score(processed_df, kmeans.labels_)
    silhouette_scores.append(score)

# --- Menyimpan visualisasi Elbow dan Silhouette Score --- #
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(K, sse, marker='o')
plt.title('Metode Elbow - Menentukan Jumlah Klaster')
plt.xlabel('Jumlah Klaster (k)')
plt.ylabel('SSE')

plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, marker='o', color='orange')
plt.title('Skor Silhouette untuk Setiap Jumlah Klaster')
plt.xlabel('Jumlah Klaster (k)')
plt.ylabel('Skor Silhouette')

plt.tight_layout()
plt.savefig('elbow_and_silhouette_scores.png')  # Simpan sebagai PNG
# plt.savefig('elbow_and_silhouette_scores.jpg')  # Jika ingin menyimpan sebagai JPG
plt.show()  # Tampilkan visualisasi
plt.close()

# Gunakan k terbaik (misalnya k=3 berdasarkan hasil visualisasi Elbow dan Silhouette)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(processed_df) # Moved this line up

# --- Menyimpan visualisasi distribusi per klaster --- #
important_features = ['age', 'bmi', 'avg_glucose_level', 'stroke']

plt.figure(figsize=(16, 10))
for i, col in enumerate(important_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=df, x='Cluster', y=col, palette='Set2') # Now 'Cluster' column exists in df
    plt.title(f'Distribusi {col} per Klaster')
    plt.xlabel('Klaster')
    plt.ylabel(col)

plt.tight_layout()
plt.savefig('distribution_per_cluster.png')  # Simpan sebagai PNG
# plt.savefig('distribution_per_cluster.jpg')  # Jika ingin menyimpan sebagai JPG
plt.show()  # Tampilkan visualisasi
plt.close()


# Rata-rata setiap klaster untuk interpretasi hasil segmentasi
print("\nRata-rata Setiap Klaster Berdasarkan Fitur Utama:\n")
print(df.groupby('Cluster').mean(numeric_only=True))

# Tugas Deep Learning

## Judul: Perbandingan Vision Transformer (ViT), Swin Transformer, dan DeiT dalam Klasifikasi Penyakit Daun Padi

---

**Author**:

- Yohanna Anzelika Sitepu (122140010)

**Semester**: Ganjil 2025  
**Institut Teknologi Sumatera (ITERA)**  
**Mata Kuliah**: IF25-40401 – Deep Learning

---

## Deskripsi Tugas

Tugas ini membandingkan tiga arsitektur Vision Transformer—ViT-B/16, Swin-Tiny, dan DeiT-Tiny—pada dataset klasifikasi penyakit daun padi. Eksperimen menggunakan strategi **transfer learning** dari bobot pretrained ImageNet-1k, dengan konfigurasi training yang identik untuk memastikan perbandingan yang adil.

**Dataset**:

- **Jumlah total**: 5.932 citra daun padi
- **Kelas**: 4 jenis penyakit
  - `Bacterialblight` (317)
  - `Blast` (281)
  - `Brownspot` (316)
  - `Tungro` (273)
- **Sumber**: Dataset custom (koleksi pribadi)
- **Distribusi**: Cukup merata (selisih maksimal ~44 gambar)
- **Lokasi**: [Google Drive – Dataset Tugas Yohanna](https://drive.google.com/drive/folders/1h5vmKgw-VAmjtCOV8PXcysJMT1OFIdD8?usp=drive_link)  
  _(Akses publik – "Anyone with the link can view")_

> Dataset **tidak disertakan di GitHub** karena ukuran (~195 MB) melebihi batas file GitHub (100 MB).

**Model yang Dibandingkan**:

- **Vision Transformer (ViT-B/16)** – transformer murni dengan attention global.
- **Swin Transformer (Swin-Tiny)** – hierarchical transformer dengan shifted window attention.
- **DeiT (DeiT-Tiny)** – ViT yang dilatih dengan knowledge distillation untuk efisiensi data.

> **Bonus nilai**: Membandingkan **3 model** + **visualisasi attention map (Grad-CAM)** → memenuhi kriteria bonus maksimal.

---

## Konfigurasi Eksperimen

| Komponen              | Nilai/Konfigurasi                                                        |
| --------------------- | ------------------------------------------------------------------------ |
| **Framework**         | PyTorch + `timm`                                                         |
| **Strategi Training** | Transfer learning (pretrained ImageNet-1k)                               |
| **Split Data**        | 70% train (4152), 10% validation (593), 20% test (1187)                  |
| **Optimizer**         | Adam                                                                     |
| **Learning Rate**     | 1e-4                                                                     |
| **Batch Size**        | 32                                                                       |
| **Epochs**            | 3                                                                        |
| **Input Size**        | 224 × 224                                                                |
| **Augmentasi**        | RandomHorizontalFlip, RandomRotation(10°)                                |
| **Preprocessing**     | Normalisasi ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) |
| **Hardware**          | Google Colab (Tesla T4 GPU, 16 GB VRAM)                                  |

> Strategi training identik untuk ketiga model guna memastikan **perbandingan yang adil**.

---

## Arsitektur Model (Ringkasan)

- **ViT-B/16**:  
  Memproses gambar sebagai urutan patch 16×16, menggunakan self-attention global. Sangat kuat pada dataset besar, namun rentan overfit pada data kecil.

- **Swin-Tiny**:  
  Menggunakan attention lokal dalam jendela bergeser (_shifted windows_) dan hierarki spasial mirip CNN. Lebih stabil dan efisien untuk tugas klasifikasi lokal.

- **DeiT-Tiny**:  
  Dilatih dengan distilasi pengetahuan dari teacher model (RegNetY), memungkinkan konvergensi cepat dan generalisasi baik meski dengan data terbatas.

---

## Hasil Eksperimen

### 🔹 Metrik Performa (Test Set: 1187 gambar)

| Model         | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
| ------------- | -------- | ----------------- | -------------- | ---------------- |
| **ViT-B/16**  | 1.00     | 1.00              | 1.00           | 1.00             |
| **Swin-Tiny** | 1.00     | 1.00              | 1.00           | 1.00             |
| **DeiT-Tiny** | 1.00     | 1.00              | 1.00           | 1.00             |

> **Confusion matrix** menunjukkan **nol kesalahan klasifikasi** untuk ketiga model.

### 🔹 Jumlah Parameter dan Ukuran Model

| Model         | Total Params | Trainable Params | Ukuran (MB) |
| ------------- | ------------ | ---------------- | ----------- |
| **ViT-B/16**  | 85,801,732   | 85,801,732       | 327.31      |
| **Swin-Tiny** | 27,522,430   | 27,522,430       | 104.99      |
| **DeiT-Tiny** | 5,525,188    | 5,525,188        | 21.08       |

### 🔹 Waktu Inferensi (Test Set: 1187 gambar)

| Model         | Inferensi per Gambar (ms) | Throughput (img/sec) |
| ------------- | ------------------------- | -------------------- |
| **ViT-B/16**  | 396.80 ± 108.02           | 2.52                 |
| **Swin-Tiny** | 148.05 ± 23.49            | 6.75                 |
| **DeiT-Tiny** | **28.42 ± 1.00**          | **35.19**            |

> **DeiT-Tiny** **14× lebih cepat** dari ViT dan **5× lebih ringan** dari Swin.

---

## Visualisasi

- **Kurva Learning**:  
  Menunjukkan konvergensi stabil dalam 3 epoch untuk ketiga model. ViT sempat menurun di Val Acc epoch 2 (0.9882), namun pulih di epoch 3.

- **Confusion Matrix**:  
  Visualisasi matriks 4×4 dengan semua prediksi benar (nol error).

- **Attention Rollout (Grad-CAM)**:  
  Overlay heatmap menunjukkan fokus model pada **area gejala penyakit** (bintik, garis, perubahan warna), bukan latar belakang — membuktikan interpretabilitas model.

---

## Analisis

1. **Ketiga model mencapai akurasi 100%** karena dataset relatif mudah dan model pretrained sangat kuat.
2. **ViT-B/16** memiliki ukuran 15× lebih besar dari DeiT dan **sangat lambat**, sehingga tidak cocok untuk aplikasi real-time.
3. **Swin-Tiny** memberikan keseimbangan antara akurasi dan efisiensi.
4. **DeiT-Tiny** adalah **pilihan optimal** untuk skenario edge/AI on-device karena kecepatan dan ukuran kecil tanpa mengorbankan akurasi.

---

## Kesimpulan dan Rekomendasi

| Use Case                       | Model Rekomendasi  |
| ------------------------------ | ------------------ |
| **Akurasi maksimal**           | Semua model (100%) |
| **Efisiensi komputasi**        | **DeiT-Tiny**      |
| **Aplikasi real-time**         | **DeiT-Tiny**      |
| **Analisis interpretabilitas** | ViT atau DeiT      |

**Saran**:

- Gunakan **DeiT-Tiny** untuk aplikasi mobile/pertanian presisi.
- Eksperimen lanjutan bisa mencakup dataset lebih besar, fine-tuning lebih lama, atau integrasi dengan MAE.

---

## Peran AI Assistant

- Membantu memperbaiki preprocessing (menambahkan **normalisasi ImageNet** dan **split val/test yang valid**).
- Membantu mengecek laporan dan README dengan struktur sistematis sesuai TOR.
- **Tidak digunakan** untuk menghasilkan hasil eksperimen atau menggantikan pemahaman konsep.

---

## Cara Menjalankan

1. **Clone repository**
   ```bash
   git clone https://github.com/YohannaAnzelika/VisionTransformer-Comparison.git
   ```
2. **Instal dependensi**
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan di Google Colab**

- Mount Google Drive:

  ```bash
  from google.colab import drive
  drive.mount('/content/drive')

  ```

- Memastikan dataset berada di:

  ```bash
  /content/drive/My Drive/Dataset_Tugas_Yohanna/archive (2).zip

  ```

- Buka file vision_transformer_comparison.ipynb dan jalankan semua cell secara berurutan.
  `vision_transformer_comparison.ipynb`

## Link Repository

- **GitHub Repository**:  
  [https://github.com/YohannaAnzelika/VisionTransformer-Comparison](https://github.com/YohannaAnzelika/VisionTransformer-Comparison)

---

## File yang Disertakan

- `vision_transformer_comparison.ipynb` – Notebook Colab lengkap
- `requirements.txt` – Daftar dependensi
- `README.md` – Dokumentasi ini
- Screenshot hasil eksperimen dan visualisasi (pada Laporan Latex)

---

## Referensi

1. Dosovitskiy, A., et al. (2021). _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_. ICLR.
2. Liu, Z., et al. (2021). _Swin Transformer: Hierarchical Vision Transformer using Shifted Windows_. ICCV.
3. Touvron, H., et al. (2021). _Training data-efficient image transformers & distillation through attention_. ICML.
4. He, K., et al. (2022). _Masked Autoencoders Are Scalable Vision Learners_. CVPR.
5. Wang, X., et al. (2023). _Vision Transformers: A Survey of Recent Advances_. IEEE Access.

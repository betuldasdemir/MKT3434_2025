
# Machine Learning GUI - Betül Daşdemir

Bu proje, öğrenci numarası **20067502** olan Betül Daşdemir tarafından geliştirilmiş kapsamlı bir grafiksel kullanıcı arayüzü (GUI) uygulamasıdır. PyQt6 ile geliştirilmiş bu uygulama, klasik makine öğrenimi, derin öğrenme ve boyut indirgeme algoritmalarını tek bir arayüzde birleştirerek kullanıcıların veri yükleme, model eğitimi, değerlendirme ve görselleştirme işlemlerini kolayca gerçekleştirmesini sağlar.

---

## 🔍 Genel Özellikler

### 📁 Veri Yönetimi
- **Veri Setleri**: Iris, Boston Housing, Breast Cancer ve özel CSV yükleme.
- **Eksik Veri İşleme**: Boş bırak, Ortalama ile doldur (Planlanmış).
- **Özellik Ölçekleme**: Yok, Standard, Min-Max, Robust.
- **Veri Bölme**: 
  - 80-20 (train-test)
  - 70-15-15, 60-20-20 (train-val-test)
  - **K-Fold çapraz doğrulama** (2–20 arası kat seçilebilir).

### 📊 Görselleştirme
- **3B Ham Veri Görselleştirme**: Seçilebilir X, Y, Z eksenleri.
- **Model Tahminleri Görselleştirme**: Gerçek vs tahmin değerleri.
- **Histogram ve renklendirme ile destekli**.
- **Metin tabanlı metrikler**: RMSE, Accuracy, Confusion Matrix vb.

---

## 🤖 Modelleme

### 1. Klasik ML
- **Regresyon**: Linear Regression
- **Sınıflandırma**: Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, KNN
- **Kümeleme**: K-Means
- **Model parametreleri**: Widget’lar ile değiştirilebilir.
- **Çapraz Doğrulama**: K-Fold doğrulama ile Accuracy, RMSE çıktıları.

### 2. Derin Öğrenme
- **Çok Katmanlı Algılayıcı (MLP)**: Dense, Dropout, Flatten gibi katmanlarla yapılandırılabilir.
- **Katman Ekleme Arayüzü**: Katman türü, aktivasyon fonksiyonu, dropout oranı vs.
- **Eğitim Parametreleri**: Batch size, Epoch, Learning rate.
- **Kayıp Fonksiyonları**:
  - Sınıflandırma için: Cross Entropy, Binary Cross Entropy, Hinge
  - Regresyon için: MSE, MAE, Huber Loss (δ ayarı ile)
- **Eğitim ilerlemesi**: Progress bar ve matplotlib grafik desteği

### 3. Boyut İndirgeme
- **PCA**: Açıklanan varyans grafiği, 2B-3B projeksiyonlar.
- **LDA**: Sınıf ayrımı için histogram/scatter.
- **K-Means**: Kümeleme + Elbow ve Silhouette metrikleri.
- **t-SNE**: Perplexity ayarı ile 2D/3D görselleştirme.
- **Tüm yöntemler sekmeler altında ayrı ayrı çalıştırılabilir ve görselleştirilebilir.**

### 4. Pekiştirmeli Öğrenme (Beta)
- **Ortamlar**: CartPole-v1, MountainCar-v0, Acrobot-v1
- **Algoritmalar**: Q-Learning, SARSA, DQN (taslak arayüz)

---

## 🔧 Kurulum

1. Python 3.8+ kurulu olmalıdır.
2. Aşağıdaki komutla bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

Alternatif olarak doğrudan:

```bash
pip install numpy pandas matplotlib PyQt6 scikit-learn tensorflow
```

3. Ana uygulamayı başlatın:

```bash
python 20067502.py
```

---

## 🖥️ Kullanım Talimatları

1. Uygulamayı başlattığınızda sol üstte veri yükleme ve ayar kısımları görünür.
2. Eksik veri ve ölçekleme seçeneklerini seçin.
3. Kaybı ayarlayıp model sekmelerinden algoritma seçin ve parametreleri belirleyin.
4. **Train** düğmesine basarak eğitimi başlatın.
5. Görselleştirme kısmından sonuçları analiz edin.

---

## 📁 Dosya Yapısı

```
├── 20067502.py        # Ana uygulama dosyası
├── README.md          # Bu belge
├── 20067502.pdf       # Rapor
└── ... (varsa örnek CSV dosyaları)
```

---

## 📌 Notlar ve Ekstra Bilgiler

- **Kod yapısı modülerdir**, her sekme bir fonksiyon içinde ayrı olarak tanımlanmıştır.
- **Exception Handling** yapılmıştır: Kullanıcı hatalarına karşı GUI içinde uyarılar verilir.
- **Kullanıcı dostu**: Widget boyutları, etiket hizalamaları ve görsel düzenlemeler optimize edilmiştir.

---

## 👩‍💻 Geliştirici

**Adı**: Betül Daşdemir  
**Öğrenci No**: 20067502  
**Kurum**: Yıldız Teknik Üniversitesi

---

## 🔓 Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.

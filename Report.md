# Lojistik Regresyon ile Kalp Hastalığı Teşhisi Projesi Raporu


**Öğrencinin adı-soyadı:** *Danial Pourrashidi Alibeiglou*
**Ders adi:** *Yapay zeka ve Makine Öğrenimine giriş*
**Öğrenci numarası:** 250121007


## Giriş

Bu proje, lojistik regresyon algoritması kullanarak kalp hastalığı teşhisi yapmayı amaçlamaktadır. Kullanılan veri seti "heart.csv" dosyası olup, hastaların yaş, cinsiyet, göğüs ağrısı tipi (cp), dinlenme kan basıncı (trestbps), kolesterol seviyesi (chol), açlık kan şekeri (fbs), dinlenme EKG sonuçları (restecg), maksimum kalp atış hızı (thalach), egzersiz kaynaklı anjina (exang), eski tepe (oldpeak), eğim (slope), büyük damar sayısı (ca), thalassemia (thal) ve hedef değişken (target: 0 - hastalık yok, 1 - hastalık var) gibi özelliklerini içermektedir. Veri seti 1025 satır ve 14 sütundan oluşmaktadır.

Proje, lojistik regresyon ile k-means kümeleme algoritmasının temel farklarını vurgulamaktadır:
- K-means unsupervised bir kümeleme algoritmasıdır; etiket (label) olmadan veri gruplarını genel olarak ayırır.
- Lojistik regresyon supervised bir sınıflandırma yöntemidir; hedef değişken (target) tanımlı olup, ikili sonuç (0 veya 1) üretir.
- Bu veri setinde lojistik regresyon, k-means'e göre daha yüksek doğruluk oranı sağlamıştır, çünkü hedef odaklı bir yaklaşım benimsemektedir.

Proje adımları: Kütüphane yükleme, veri analizi, model eğitimi, test ve doğruluk hesaplama.

## Yöntem

### 1. Kütüphane Yükleme ve Veri Analizi
Kullanılan kütüphaneler:
- Pandas: Veri çerçeveleri için.
- NumPy: Sayısal işlemler için.
- Scikit-learn: Lojistik regresyon modeli (LogisticRegression), doğruluk skoru (accuracy_score), veri bölme (train_test_split) ve ölçeklendirme (StandardScaler).

Veri seti pandas ile okunmuş ve analiz edilmiştir:
- `df.info()`: Veri türleri ve boş değerler kontrolü (tüm sütunlar dolu, 1 float ve 13 int tipi).
- `df.columns`: Sütun isimleri listelenmiştir.

Veri ön işleme:
- Özellik matrisi (X): Target sütunu hariç tüm sütunlar.
- Hedef vektör (y): Target sütunu.
- Veri bölme: %80 eğitim (train), %20 test (test_size=0.2, random_state=36).
- Ölçeklendirme: StandardScaler ile özellikler standartlaştırılmış, modelin daha tutarlı tahmin yapması sağlanmıştır.

### 2. Model Eğitimi ve Testi
Model: LogisticRegression() nesnesi oluşturulmuş ve eğitim verisi ile eğitilmiştir (`model.fit(X_train, y_train)`).

Tahmin: Test verisi üzerinde tahmin yapılmış (`y_pred = model.predict(X_test)`).

Sonuç Tablosu: Test verisi orijinal hale getirilmiş (`scaler.inverse_transform`), gerçek ve tahmin değerleri bir DataFrame'e eklenmiştir. Örnek 10 satır:

| age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal | gercek | predict |
|-----|-----|----|----------|------|-----|---------|---------|-------|---------|-------|----|------|--------|---------|
| 70  | 1   | 1  | 156      | 245  | 0   | 0       | 143     | 0     | 0.0     | 2     | 0  | 2    | 1      | 1       |
| 57  | 1   | 2  | 150      | 168  | 0   | 1       | 174     | 0     | 1.6     | 2     | 0  | 2    | 1      | 1       |
| 55  | 1   | 0  | 160      | 289  | 0   | 0       | 145     | 1     | 0.8     | 1     | 1  | 3    | 0      | 0       |
| 59  | 1   | 0  | 140      | 177  | 0   | 1       | 162     | 1     | 0.0     | 2     | 1  | 3    | 0      | 0       |
| 49  | 1   | 2  | 118      | 149  | 0   | 0       | 126     | 0     | 0.8     | 2     | 3  | 2    | 0      | 0       |
| 56  | 1   | 3  | 120      | 193  | 0   | 0       | 162     | 0     | 1.9     | 1     | 0  | 3    | 1      | 1       |
| 43  | 1   | 0  | 115      | 303  | 0   | 1       | 181     | 0     | 1.2     | 1     | 0  | 2    | 1      | 1       |
| 50  | 1   | 2  | 129      | 196  | 0   | 1       | 163     | 0     | 0.0     | 2     | 0  | 2    | 1      | 1       |
| 62  | 1   | 2  | 130      | 231  | 0   | 1       | 146     | 0     | 1.8     | 1     | 3  | 3    | 1      | 0       |
| 58  | 1   | 0  | 128      | 216  | 0   | 0       | 131     | 1     | 2.2     | 1     | 3  | 3    | 0      | 0       |

Doğruluk Hesabı: `accuracy_score(y_test, y_pred)` ile hesaplanmış, sonuç %82.93'tür.

## Sonuç ve Değerlendirme

Lojistik regresyon modeli, test verisinde %82.93 doğruluk oranı elde etmiştir. Bu, k-means'e göre daha yüksek bir performans gösterir, çünkü lojistik regresyon hedef değişkeni doğrudan kullanarak ikili sınıflandırma yapar. K-means genel kümeleme sağlarken, bu projede etiketli veri ile lojistik regresyonun üstünlüğü belirgindir.

Danial Pourrashidi Alibeiglou,

Thanks to Enes Gorgulu.

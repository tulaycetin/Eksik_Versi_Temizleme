# Eksik Veri İmputation ve Lineer Regresyon Analizi Projesi

Bu projede, eksik verileri doldurmak için üç farklı yöntem kullanılmış ve her biriyle oluşturulan veri setleri üzerinde lineer regresyon analizi gerçekleştirilmiştir. Amaç; Maximum Likelihood Estimation (MLE), Multiple Imputation by Chained Equations (MICE) ve Expectation-Maximization (EM) yöntemleriyle imputation yapılmış veri setlerinin regresyon performansını karşılaştırarak hangi yöntemin daha iyi sonuç verdiğini belirlemektir.

## Kullanılan İmputation Yöntemleri

1. **Maximum Likelihood Estimation (MLE):**  
   - Verinin normal dağılım varsayımına dayanarak eksik değerleri doldurur.
   - Örnek olarak, orijinal verinin parametreleri (mu, sigma) optimize edilip, eksik değerler bu dağılımdan örneklenir.

2. **Multiple Imputation by Chained Equations (MICE):**  
   - Eksik değerleri, diğer değişkenlerle olan ilişkileri göz önüne alarak iteratif olarak doldurur.
   - Değişkenler arası bağımlılıkları koruyarak, birden fazla eksik veri durumunda daha güvenilir sonuçlar üretebilir.

3. **Expectation-Maximization (EM) Algoritması:**  
   - Eksik verileri, Gaussian Mixture Model gibi istatistiksel modeller üzerinden tahmin eder.
   - Genellikle tek bileşenli bir model kullanılarak eksik değerler doldurulur; bu yöntem de normal dağılım varsayımına benzer şekilde çalışır.

## Proje Akışı

1. **Veri Hazırlığı:**  
   - Orijinal veri setinin bir kopyası alınır ve eksik veriler yukarıda belirtilen üç yöntem kullanılarak doldurulur.
   - Kategorik sütunlarda yüksek kardinaliteli (100’den fazla benzersiz değer içeren) sütunlar, gerekirse modellemeye dahil edilmeden kaldırılır.

2. **Veri Seti Oluşturulması:**  
   - Her imputation yöntemiyle elde edilen veri setinde, hedef değişken (örneğin, `Win_Prob`) ve özellikler ayrılır.
   - Gerektiğinde tarih veya zaman gibi sütunlar veri setinden çıkarılır.

3. **Lineer Regresyon Analizi:**  
   - Her imputation yöntemiyle oluşturulan veri setinde ayrı ayrı lineer regresyon modeli eğitilir.
   - Modelin performansı; Mean Squared Error (MSE), Mean Absolute Error (MAE) ve R² Score gibi metriklerle değerlendirilir.
   - Ayrıca, belirli bir eşik değeri kullanılarak yapılan binary sınıflandırma sonucu Confusion Matrix, Accuracy ve Classification Report ile incelenir.

4. **Sonuçların Karşılaştırılması:**  
   - Üç yöntemin regresyon ve sınıflandırma metrikleri karşılaştırılır.
   - Genellikle, MICE yöntemi değişkenler arası ilişkiyi daha iyi koruduğu için daha sağlam sonuçlar üretebilir; ancak veri setinin özelliklerine bağlı olarak diğer yöntemler de uygun sonuç verebilir.

## Örnek Kod Parçacıkları

Her yönteme ait imputation ve model eğitim süreci aşağıdaki gibi özetlenebilir:

```python
# MLE İle Imputation
df_mle = mle_imputation(df)
X_mle, y_mle = prepare_data(df_mle)
print("\n--- Maximum Likelihood Model Eğitimi ---")
linear_model(X_mle, y_mle)

# MICE İle Imputation
df_mice = mice_imputation(df)
X_mice, y_mice = prepare_data(df_mice, target_column='Win_Prob', drop_columns=[])
print("\n--- MICE Model Eğitimi ---")
linear_model(X_mice, y_mice)

# EM Algoritması ile Imputation
df_em = em_imputation(df)
X_em, y_em = prepare_data(df_em)
print("\n--- EM Model Eğitimi ---")
linear_model(X_em, y_em)

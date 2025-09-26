# Akbank-Derin-Ogrenmeye-Giris-Bootcamp

# GİRİŞ

Bu repo, Global AI Hub Python  Derin Öğrenmeye Giriş Bootcamp ı sürecinde bitirme projesini sunmak amacıyla oluşturulmuştur. 

# PNÖMONİ NEDİR?

<img width="857" height="482" alt="image" src="https://github.com/user-attachments/assets/4566816e-463b-41a5-a5bf-21c1c7f768b9" />

Projenin amacından söz etmeden önce inceleyeceğimiz hastalık hakkında genel bir bilgi sahibi olalım. 
Pnömoni, öncelikle alveoller olarak bilinen küçük hava keseciklerini etkileyen bir akciğer iltihaplanması durumudur. Belirtiler genellikle verimli veya kuru öksürük, göğüs ağrısı, ateş ve nefes darlığı gibi bir kombinasyonu içerir. Durumun şiddeti değişkenlik gösterebilir. Pnömoni genellikle virüsler veya bakterilerle enfeksiyon nedeniyle ortaya çıkar; daha az yaygın olarak diğer mikroorganizmalar, belirli ilaçlar veya otoimmün hastalıklar gibi durumlar da sebep olabilir. Risk faktörleri arasında kistik fibroz, kronik obstrüktif akciğer hastalığı (KOAH), astım, diyabet, kalp yetmezliği, sigara öyküsü, inme sonrası öksürememe gibi durumlar ve zayıf bağışıklık sistemi bulunur. Tanı genellikle belirtiler ve fizik muayene temel alınarak konur. Göğüs röntgeni, kan testleri ve balgam kültürü tanıyı doğrulamaya yardımcı olabilir.

# PROJENİN AMACI

Bu çalışmanın amacı, hem yetişkin hem de çocuk göğüs röntgenlerini pnömoni ve normal olarak sınıflandırabilen bir derin öğrenme modeli geliştirmek ve pnömoni hastalarının teşhis sürecine katkı sağlamaktır. Literatürdeki mevcut derin öğrenme çalışmaları genellikle yalnızca yetişkin veya yalnızca çocuk röntgenleri üzerine odaklanmakta ve her iki hasta grubunu aynı anda kapsayan modeller sınırlı sayıdadır. Bu bağlamda, bu çalışmada her iki hasta grubunun röntgen verileri modelin girdisi olarak kullanılarak, hem yetişkin hem de çocuk hastalara hizmet verebilecek bütüncül bir sınıflandırma sistemi geliştirilmesi hedeflenmektedir.

# VERİ SETİ HAKKINDA GENEL BİLGİ
Projede iki farklı Kaggle veri seti birleştirilerek model eğitimi yapılmıştır.
### 1. [Chest X-Ray Covid-19 & Pneumonia Veri Seti](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)  


COVID-19 (coronavirus disease 2019), koronavirüslerin bir türü olan şiddetli akut solunum yolu sendromu koronavirüs 2 (SARS-CoV-2) tarafından neden olunan bulaşıcı bir hastalıktır. İlk vakalar Aralık 2019’un sonlarında Çin’in Wuhan kentinde görülmüş ve daha sonra küresel ölçekte yayılmıştır. Mevcut salgın, Dünya Sağlık Örgütü (DSÖ) tarafından 11 Mart 2020 tarihinde resmî olarak pandemi ilan edilmiştir. Günümüzde COVID-19’un teşhisinde ters transkripsiyon polimeraz zincir reaksiyonu (RT-PCR) kullanılmaktadır. Röntgen cihazlarının yaygın olarak bulunması ve hızlı görüntü sağlayabilmesi nedeniyle, göğüs röntgenleri COVID-19’un erken teşhisinde oldukça faydalı bir yöntemdir.

Veri seti, iki klasörde (train ve test) organize edilmiştir. Hem train hem de test klasörleri, üç alt klasör (COVID19, PNEUMONIA, NORMAL) içermektedir. Veri setinde toplam 6.432 göğüs röntgeni bulunmaktadır ve test verisi toplam görüntülerin %20’sini kapsamaktadır. Bu çalışmada yalnızca PNEUMONIA, NORMAL kalsörlerindeki veriler kullanılacaktır.


### 2. [Chest X-Ray & Pneumonia Veri Seti](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data)

Veri seti, üç klasör (train, test, validation) şeklinde organize edilmiş olup, her bir görüntü kategorisi (Pnömoni/Normal) için alt klasörler içermektedir. Toplamda 5.863 göğüs röntgeni (JPEG formatında) bulunmakta ve bu görüntüler iki kategoriye (Pnömoni ve Normal) ayrılmaktadır.

Veri seti, üç klasör (train, test, val) şeklinde organize edilmiştir ve her görüntü kategorisi (Pnömoni/Normal) için alt klasörler içermektedir. Toplamda 5.863 göğüs röntgeni (JPEG formatında) ve iki kategori (Pnömoni/Normal) bulunmaktadır. Göğüs röntgenleri (antero-posterior), Çin’in Guangzhou Kadın ve Çocuk Sağlığı Merkezi’ne başvuran, 1 ila 5 yaş arasındaki pediatrik hastaların geriye dönük kohortlarından seçilmiştir. Tüm görüntüler, hastaların rutin klinik bakım süreçleri sırasında elde edilmiştir. Analiz sürecinde, düşük kaliteli veya okunamayan taramalar çıkarılarak tüm göğüs röntgenleri öncelikle kalite kontrolünden geçirilmiştir. Görüntülere ait tanılar, yapay zekâ sistemi için eğitime dâhil edilmeden önce iki uzman hekim tarafından değerlendirilmiş; olası derecelendirme hatalarını önlemek amacıyla değerlendirme seti ayrıca üçüncü bir uzman hekim tarafından da kontrol edilmiştir.

Tabloda görüldüğü üzere pnömoni ve normal sınıfları arasındaki veri dağılımı dengesizdir. Bu nedenle, iki farklı veri seti birleştirilerek hem sınıflar arası denge sağlanması hem de veri çeşitliliğinin artırılması hedeflenmiştir. Böylece modelin genel performansının ve genellenebilirliğinin nasıl etkileneceği incelenmiştir.

## Veri Seti Dağılımı

| Sınıf      | Chest X-Ray & Pneumonia (Train) | Chest X-Ray & Pneumonia (Validation) | Chest X-Ray & Pneumonia (Test) | Covid-19 & Pneumonia (Train) | Covid-19 & Pneumonia (Test) |
|------------|---------------------------------|--------------------------------------|--------------------------------|-------------------------------|------------------------------|
| **PNEUMONIA** | 3875                            | 8                                    | 390                            | 3418                          | 855                          |
| **NORMAL**   | 1341                            | 8                                    | 234                            | 1266                          | 317                          |

### VERİ SETİ OLUŞTURMA STRATEJİSİ

1. Chest X-Ray & Pneumonia Veri Seti Validation veri setinin Train veri setinden resim alınarak arttırılması
2. Chest X-Ray Covid-19 & Pneumonia veri setinin Train veri setinden resim alınarak validation datasının oluşturulması
3. Pnömoni ve Normal sınıf dağılımı, train setindeki oranla aynı olacak şekilde validation ve test datası oluşturulmuştur. Yani iki veri setide sınıf dağılımı açısından dengesizdir. Bu dengesizlik, veri arttırma yöntemi kullanılarak train setinden modelin en iyi derecede öğrenmesinin sağlanması hedeflenmektedir.
4. Hem çocuk hem yetişkin göğüs görüntüleri kullanılarak aynı modelin iki farklı çeşitte görüntü girdisi ile pnömoni hastalığının teşhis edilmesi amaçlanmıştır.


Verilerin dağılımı yukarıda belirttiğim şekilde yapılmış olup, veri dağılımları aşağıdaki tabloda belirtilmiştir. 

<img width="1850" height="318" alt="image" src="https://github.com/user-attachments/assets/84ea675e-c722-4005-a02f-3efcaf0f9bf3" />

5. İki veri setinin train, validation ve test verisi için de mantıklı bir stateji izlenmiştir. Train seti birleştirilirken shuffle işlemi uygulanmıştır. Bunun uygulanmasındaki temel amaç, modelin veri setinin örüntüsünü öğrenirken genelleme gücünü artırmak, sıralı bağımlılıklardan kaçınmak ve  overfitting’i azaltmaktır.<br>
Ayrıca iki veri setine ait test ve validation verileri birleştirilirken sıralama önce  Chest X-Ray & Pneumonia veri setinin test ve validation datası sonra Chest X-Ray Covid-19 & Pneumonia olacak şekilde bırakılmıştır. Aynı model aynı veri üzerinde farklı sonuçlar üretilmesini engellemektir. Bu şekilde modelin hem yetişkin hem de çocuk göğüs verilerinden belirli kalıplarını öğrenip ikisinde de doğru sınıflandırma yapması hedefleniyor.
6. Birleştirilmiş veri seti Kaggle notebook üzerinde output kısmına combined_dataset.npz adıyla kaydedilmiştir.


# 3.	KULLANILAN YÖNTEMLER

 ## 1. Veri Ön İşleme: 

  **CLAHE (Contrast Limited Adaptive Histogram Equalization):** ile görüntü kontrastı artırılır. Böylece akciğer dokusundaki küçük detaylar ve opasiteler daha belirgin hâle gelir.
  **img_size** Modelin kolay bir şekilde öğrenmesi için tüm görüntüler 150x150 piksellik boyuta indirgenmiştir. Genel kabul sağlık verilerinde 224x224 kullanılmasıdır.
  ## 2. Veri Arttırma(Data Augmentation)


Derin öğrenme modellerinde başarının en önemli faktörlerinden biri, eğitim için kullanılan verinin çeşitliliği ve miktarıdır. Ancak gerçek dünyada geniş ve dengeli veri setlerine ulaşmak çoğu zaman mümkün değildir. Bu noktada **veri arttırma (data augmentation)** teknikleri, mevcut veriler üzerinde çeşitli dönüşümler uygulayarak yapay örnekler üretmekte ve modelin genelleme kabiliyetini artırmaktadır. Keras  kütüphanesinin ImageDataGenerator yöntemi kullanılmıştır. Veri arttırma modelleme öncesi sadece train setine uygulanmıştır.

Aşağıda, çalışmada kullanılan veri arttırma parametreleri ve işlevleri özetlenmiştir:

- **`rotation_range=30`**  
  Görüntüler rastgele **-30° ile +30°** arasında döndürülür. Böylece model, farklı açılarda elde edilmiş görsellere karşı duyarsız hale gelir.  

- **`zoom_range=0.2`**  
  Görüntüler rastgele **%20 oranında yakınlaştırılır veya uzaklaştırılır**. Bu sayede model, nesnelerin farklı uzaklıklardan çekilmiş varyasyonlarını öğrenebilir.  

- **`width_shift_range=0.1`**  
  Görüntüler yatay eksende **%10 oranında kaydırılır**. Böylece farklı yatay konumlara karşı modelin dayanıklılığı artırılır.  

- **`height_shift_range=0.1`**  
  Görüntüler dikey eksende **%10 oranında kaydırılır**. Bu yöntem, farklı hizalamalara karşı modelin esnekliğini artırır.  

- **`horizontal_flip=True`**  
  Görüntüler yatay eksende ayna görüntüsü alınarak çevrilir. Bu dönüşüm, özellikle sağ-sol simetriye sahip verilerde modelin daha güçlü genelleme yapmasına katkı sağlar.  

- **`vertical_flip=True`**  
  Görüntüler dikey eksende ters çevrilir. Ancak, özellikle tıbbi görüntü analizinde anatomik gerçekliğe uymadığından bu yöntem genellikle tercih edilmez. Daha çok nesne tanıma veya doğa görüntülerinde kullanılmaktadır.  

### Sonuç
Veri arttırma teknikleri sayesinde model, yalnızca eğitim verilerini ezberlemekten öteye geçerek **farklı varyasyonlara uyum sağlayabilen**, daha **genelleştirilebilir ve dayanıklı (robust)** bir yapıya kavuşmaktadır. Bu yöntem, özellikle tıbbi görüntü analizlerinde veri kısıtlılığını aşmak ve model performansını artırmak için kritik bir rol oynamaktadır. Bizim modelimizde de train seti genellenebilirlik kapasitesi iyi olduğu sonuçlarla gösterilmiştir.
## 3. Modelleme(CNN)
### Toplam Katman Sayısı

5 (Conv2D) + 5 (BatchNorm) + 5 (MaxPool) + 4 (Dropout) + 3 (Flatten/Dense) = 22 katman


### Model Yapısı ve Teknik Terimler

Bu çalışmada göğüs röntgeni verilerinin sınıflandırılması amacıyla derin öğrenme tabanlı bir **Convolutional Neural Network (CNN)** modeli tasarlanmıştır. Modelin yapısı ve kullanılan teknikler aşağıda açıklanmaktadır:

-**Epoch sayısı:** 12 olarak seçilmiştir. 

- **Conv2D Katmanları:** Görüntülerin uzamsal özelliklerini öğrenmek amacıyla evrişimsel filtreler kullanılmıştır. Bu filtreler, akciğer röntgenlerindeki farklı dokusal ve yapısal paternleri (ör. pnömoni kaynaklı yoğunluklar) otomatik olarak yakalayabilmektedir.

- **Batch Normalization:** Her katmandan sonra uygulanan bu yöntem, aktivasyon dağılımlarını normalize ederek eğitimi hızlandırmakta ve aşırı öğrenme (overfitting) riskini azaltmaktadır.

- **MaxPooling Katmanları:** Uzamsal boyutlar küçültülerek modelin gereksiz karmaşıklığı azaltılmış, aynı zamanda daha özet özelliklerin çıkarılması sağlanmıştır. Bu yaklaşım, özellikle yüksek çözünürlüklü röntgen görüntülerinde verimlilik sağlamaktadır.

- **Dropout:** Belirli oranlarda nöronların eğitim sırasında rastgele devre dışı bırakılması, modelin belirli özelliklere aşırı bağımlı hale gelmesini engellemekte ve genelleme gücünü artırmaktadır.

- **Flatten ve Dense Katmanları:** Evrişimsel katmanlardan elde edilen özellikler düzleştirilerek tam bağlantılı katmanlara aktarılmıştır. Bu yapı, öğrenilen yüksek seviyeli özelliklerin sınıflandırma amacıyla bir araya getirilmesini sağlamaktadır.

**Stride (Adım Boyutu):** Evrişim filtresinin görüntü üzerinde kayma miktarını ifade eder. Örn: `stride=1` filtreyi bir piksel kaydırırken, `stride=2` iki piksel kaydırır. Stride büyüdükçe çıkış boyutu küçülür ve hesaplama hızı artar, ancak özellik ayrıntısı bir miktar azalır. Bu parametre, modelin hem verimli hem de anlamlı özellikler öğrenmesini sağlar.

- **Sigmoid Aktivasyon:** Çıkış katmanında ikili sınıflandırma (pnömoni / normal) yapılabilmesi için sigmoid fonksiyonu tercih edilmiştir.

### Modelin Derlenmesi

- **Optimizer – RMSprop:** Röntgen görüntüleri gibi karmaşık ve yüksek boyutlu verilerde parametre güncellemelerini daha dengeli yapmak için tercih edilmiştir. RMSprop, öğrenme oranını her parametre için uyarlayarak hızlı ve istikrarlı bir öğrenme süreci sağlar.  

- **Loss – Binary Crossentropy:** İkili sınıflandırma (pnömoni / normal) problemi için en uygun kayıp fonksiyonudur. Modelin tahminleri ile gerçek etiketler arasındaki farkı logaritmik olasılık üzerinden ölçmektedir.  

- **Metrics – Accuracy:** Sağlık verilerinde model başarısını yorumlamak için en anlaşılır metriklerden biridir. Doğru sınıflandırılan örneklerin tüm örneklere oranını vererek modelin genel doğruluğunu göstermektedir.  

### Katman Sayısının Fazla Olma Gerekçesi
Göğüs röntgenleri yüksek düzeyde karmaşık ve çok ince yapısal farklılıklar barındırmaktadır. Bu nedenle daha fazla katman kullanılarak, düşük seviyeli kenar ve dokulardan başlayıp daha yüksek seviyeli soyut özelliklere kadar derin bir temsil elde edilmiştir. Böylece model, normal ve pnömonili akciğerler arasındaki ayrımı daha güvenilir şekilde yapabilmektedir.

### Sonuç
Kullanılan derin öğrenme mimarisi, medikal görüntülerde görülen karmaşık paternleri yakalayabilmek, genelleme kabiliyetini artırmak ve aşırı öğrenmeyi engellemek amacıyla seçilmiştir. Bu yaklaşım sayesinde, göğüs röntgenlerinden pnömoni teşhisinde yüksek doğruluk elde edilmesi hedeflenmiştir.

## 4. MODEL DEĞERLENDİRME TEKNİKLERİ: 
Accuracy, Loss grafikleri + Confusion Matrix 
Grad-CAM (Gradient-weighted Class Activation Mapping)
<img width="737" height="430" alt="image" src="https://github.com/user-attachments/assets/0fcba7e2-8061-4d5f-8f1b-906d7c1d5110" />

## 6. Hiperparametre Optimizasyonu: 
Bir önceki modelden farklı olarak sondaki katman azaltıldı.
early_stop eklenmiştir. **Early stopping**, derin öğrenme eğitiminde sık kullanılan bir regularization (düzenleme) tekniğidir ve temel amacı modelin overfitting yapmasını önlemektir.
Düzleştirme katmanında dropout değeri 0.3 e yükseltilmiştir. Bunun amacı, Modelin belli nöronlara aşırı bağımlı olmasını önlemek ve genelleme yeteneğini artırmak.


# ELDE EDİLEN SONUÇLAR

<img width="408" height="394" alt="image" src="https://github.com/user-attachments/assets/00dcd12e-dfa4-4a87-b552-f84b430ed5ba" />

<img width="848" height="150" alt="image" src="https://github.com/user-attachments/assets/502af393-a7df-4815-b320-b288a7c8833d" />

<img width="841" height="257" alt="image" src="https://github.com/user-attachments/assets/98a3813b-4663-48ec-bf4b-35a4963a414a" />


# GELECEK ÇALIŞMALAR İLE İLGİLİ ÖNERİLER¶

## Mevcut Durum ve Sınırlamalar¶

Eğitim veri setindeki sınıf dengesizlikleri, özellikle test setinde model doğruluğunu etkilemiştir.
Farklı yaş gruplarına ait göğüs röntgenleri (yetişkin vs. çocuk) görsel dağılım ve özellik bakımından farklılık gösterir. Bu durum, tek bir modelin her iki dağılımı aynı anda öğrenmesini zorlaştırabilir.
Gelecek Çalışmalar için Öneriler

##Veri Setinin Çeşitlendirilmesi:

Eğitim setine daha fazla örnek eklenmesi ve sınıf dağılımının dengelenmesi, modelin genelleme kapasitesini artıracaktır.
## Transfer Learning ve Fine-Tuning:

Öncelikle yetişkin röntgenleri ile ön eğitim (pre-training) yapılabilir.
Daha sonra model, çocuk verileri ile fine-tuning sürecine tabi tutulabilir.
Bu yaklaşım sayesinde model, iki farklı dağılımdaki görselleri ayrı ayrı öğrenerek daha doğru sınıflama gerçekleştirebilir.
Modelin Genelleme Yeteneğinin Artırılması:

## Veri augmentasyonu (dönme, ölçekleme, parlaklık değişimi gibi) ve düzenleme teknikleri (dropout, early stopping) ile overfitting riski azaltılabilir.
Sınıf dengesizlikleri ve yaş grubu farklılıkları göz önünde bulundurularak performans metrikleri ayrı ayrı raporlanabilir.
# SONUÇ

Bu model, hem yetişkin hem çocuk göğüs röntgenlerinde pnömoni tespitinde başarılı bir başlangıç sağlamaktadır. Ancak, veri çeşitlendirmesi, transfer learning ve fine-tuning uygulamaları ile modelin doğruluk ve genelleme kapasitesi ileri çalışmalarda artırılabilir.

# Kaggle Linki [MODELLEME (CNN) - Kaggle Notebook](https://www.kaggle.com/code/zdengltekin/pneumonia-detection-using-cnn-combining2-dataset/notebook#MODELLEME-(CNN))






   








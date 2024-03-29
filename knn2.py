import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Datasetleri oku
friends_df = pd.read_csv(r'D:\Ytü Dersler\Bilgiye Erişim ve Arama Motorları\Proje\archive\friends_table.csv')
user_df = pd.read_csv(r'D:\Ytü Dersler\Bilgiye Erişim ve Arama Motorları\Proje\archive\userID_table.csv')

# Friend 1 ile UserID'yi birleştir
merged_df = pd.merge(friends_df, user_df, left_on='Friend 1', right_on='UserID', how='inner')

# Her bir kullanıcı için diğer kullanıcılarla olan ortak arkadaş sayılarını hesapla
user_ids = merged_df['UserID'].unique()
common_friends_matrix = []

for user_id in user_ids:
    common_friends = [len(set(merged_df[merged_df['UserID'] == user_id]['Friend 2']).intersection(set(merged_df[merged_df['UserID'] == other_user]['Friend 2']))) for other_user in user_ids]
    common_friends_matrix.append(common_friends)

# Ortak arkadaş sayılarını içeren bir DataFrame oluştur
common_friends_df = pd.DataFrame(common_friends_matrix, index=user_ids, columns=user_ids)

# Kullanıcıların yaşlarını içeren bir DataFrame oluştur
user_age_df = user_df.set_index('UserID')['Age']

# Kullanıcıdan giriş al
user_id = int(input("Arkadaşlık önerileri almak istediğiniz kullanıcının numarasını girin (Programdan çıkmak için 0'a basın): "))

while user_id != 0:
    # Kullanıcının yaşını al
    user_age = user_age_df.loc[user_id]

    # Kullanıcılar arasındaki uzaklıkları hesapla
    knn_model = NearestNeighbors(n_neighbors=3, metric='manhattan')
    flattened_features = user_age_df.values.reshape(-1, 1)
    knn_model.fit(flattened_features)

    distances, indices = knn_model.kneighbors([[user_age]])

    # Kullanıcının kümesindeki diğer kullanıcıları öneri olarak al
    cluster_users = [user for user in indices.flatten() if user != user_id]

    # Kullanıcıların isim ve soyisimlerini içeren bir DataFrame oluştur
    user_info_df = user_df.set_index('UserID')

    # Öneri listesini yazdır
    if cluster_users:
        print(f"{user_id}. kullanıcının Arkadaşlık Önerileri:")
        for onerilen_kullanici in cluster_users:
            surname = user_info_df.loc[onerilen_kullanici, 'Surname']
            name = user_info_df.loc[onerilen_kullanici, 'Name']
            print(f"({name} {surname})")
    else:
        print(f"{user_id}. kullanıcının Arkadaşlık Önerisi Bulunamadı.")

    # Kullanıcıdan tekrar giriş al
    user_id = int(input("Arkadaşlık önerileri almak istediğiniz kullanıcının numarasını girin (Programdan çıkmak için 0'a basın): "))

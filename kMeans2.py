import pandas as pd
from sklearn.cluster import KMeans

#isimlerle birlikte K-Means

# Datasetleri oku
friends_df = pd.read_csv(r'D:\Proje\archive\friends_table.csv')
user_df = pd.read_csv(r'D:\Proje\archive\userID_table.csv')

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

# K-Means kümeleme uygula
kmeans = KMeans(n_clusters=3, random_state=42)
common_friends_df['Cluster'] = kmeans.fit_predict(common_friends_df)

# Kullanıcıdan giriş al
user_id = int(input("Arkadaşlık önerileri almak istediğiniz kullanıcının numarasını girin: "))

# Kullanıcının kendi kümesindeki diğer kullanıcılara ait ortak arkadaş sayılarını al
user_cluster = common_friends_df.loc[user_id, 'Cluster']

# Kullanıcının kümesindeki diğer kullanıcıları öneri olarak al
cluster_users = common_friends_df[common_friends_df['Cluster'] == user_cluster].index.tolist()
cluster_users.remove(user_id)  # Kullanıcıyı öneri listesinden çıkar


# Ortak arkadaş sayısına göre sırala
cluster_users = sorted(cluster_users, key=lambda x: common_friends_df.loc[user_id, x], reverse=True)

# 0 ortak arkadaş sayısına sahip olanları filtrele
cluster_users = [user for user in cluster_users if common_friends_df.loc[user_id, user] > 0]

# Sadece belirli bir sayıda öneri almak için listenin ilk 4 elemanını al
num_recommendations = 4
cluster_users = cluster_users[:num_recommendations]

# Kullanıcıların isim ve soyisimlerini içeren bir DataFrame oluştur
user_info_df = user_df.set_index('UserID')

# Öneri listesini yazdır
if cluster_users:
    print(f"{user_id}. kullanıcının Arkadaşlık Önerileri:")
    for onerilen_kullanici in cluster_users:
        surname = user_info_df.loc[onerilen_kullanici, 'Surname']
        name = user_info_df.loc[onerilen_kullanici, 'Name']
        print(f"({name} {surname}) ")
else:
    print(f"{user_id}. kullanıcının Arkadaşlık Önerisi Bulunamadı.")


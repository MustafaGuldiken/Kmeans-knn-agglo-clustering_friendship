import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Verileri oku
friends_data = pd.read_csv("D:/Proje/archive/friends_table.csv")
user_data = pd.read_csv("D:/Proje/archive/userID_table.csv")
reactions_data = pd.read_csv("D:/Proje/archive/reactions_table_m.csv")

# İki veri setini birleştir
merged_data = pd.merge(friends_data, user_data, left_on='Friend 1', right_on='UserID', how='inner')
merged_data = pd.merge(merged_data, reactions_data, left_on='Friend 1', right_on='User', how='inner')

# Kullanılacak özellikleri seç
features = merged_data[['Friend 1', 'Friend 2', 'Age', 'Reaction Type', 'Name', 'Surname']]

# Verileri ölçeklendir
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features[['Friend 1', 'Friend 2', 'Age', 'Reaction Type']])

# Agglomerative clustering modelini oluştur
cluster_model = AgglomerativeClustering(n_clusters=100, affinity='euclidean', linkage='ward')

# Modeli eğit ve küme etiketlerini al
features['Cluster'] = cluster_model.fit_predict(scaled_features)

# Öneri sayısı
max_recommendations = 4

# Önerileri göstermek için kullanıcı ID'sini girelim
user_id = int(input("Önerileri görmek için bir UserID girin (0 çıkış yapmak için): "))

while user_id != 0:
    # Kullanıcının ait olduğu küme
    user_cluster = features.loc[features['Friend 1'] == user_id, 'Cluster'].values[0]

    # Aynı kümedeki diğer kullanıcıları bul
    cluster_users = features.loc[features['Cluster'] == user_cluster, ['Friend 1', 'Name', 'Surname']].drop_duplicates()

    # Kullanıcının arkadaşlarını ve kümedeki diğer kullanıcıları göster
    recommendations = cluster_users[cluster_users['Friend 1'] != user_id].head(max_recommendations)
    
    # Öneri sonuçlarını liste şeklinde göster
    print(f"\nÖneriler (Küme {user_cluster}):")
    for index, row in recommendations.iterrows():
        print(f"{row['Friend 1']} - {row['Name']} {row['Surname']}")

    # Yeniden kullanıcı ID'si al
    user_id = int(input("Önerileri görmek için bir UserID girin (0 çıkış yapmak için): "))

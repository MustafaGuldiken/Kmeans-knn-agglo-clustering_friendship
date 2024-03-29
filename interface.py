import time
import customtkinter
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("BEAM")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((1, 2, 3), weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Algoritmalar", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.agglom_bf_2, text="Agglomerative")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.kmeans_analy, text="KMeans")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.knn2_bf_2, text="KNN2")
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 20))

        self.sidebar_frame02 = customtkinter.CTkFrame(self, width=900, corner_radius=0)
        self.sidebar_frame02.grid(row=0, column=1, rowspan=4, sticky="nsew")
        self.sidebar_frame02.grid_rowconfigure(1, weight=1)
        self.logo_label02 = customtkinter.CTkLabel(self.sidebar_frame02, text="Oneriler", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label02.grid(row=1, column=1, padx=20, pady=(10, 10))
        #self.scaling_label2 = customtkinter.CTkLabel(self.sidebar_frame, text="Agglomerative:", anchor="w")

        self.logo_label03 = customtkinter.CTkLabel(self.sidebar_frame02, text="Agglomerative", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label03.grid(row=0, column=2, padx=10, pady=(10, 0))
        self.logo_label03_1 = customtkinter.CTkLabel(self.sidebar_frame02, text="Sure:", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label03_1.grid(row=2, column=2, padx=10, pady=(10, 0))
        self.textbox1 = customtkinter.CTkTextbox(self.sidebar_frame02, width=250, height=500)
        self.textbox1.grid(row=1, column=2, padx=(10, 0), pady=(20, 10), sticky="nsew")
        self.logo_label04 = customtkinter.CTkLabel(self.sidebar_frame02, text="KNN", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label04.grid(row=0, column=3, padx=10, pady=(10, 0))
        self.textbox2 = customtkinter.CTkTextbox(self.sidebar_frame02, width=250, height=500)
        self.textbox2.grid(row=1, column=3, padx=(10, 0), pady=(20, 10), sticky="nsew")
        self.logo_label04_1 = customtkinter.CTkLabel(self.sidebar_frame02, text="Sure:", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label04_1.grid(row=2, column=3, padx=10, pady=(10, 0))
        self.logo_label05 = customtkinter.CTkLabel(self.sidebar_frame02, text="KMeans", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label05.grid(row=0, column=4, padx=10, pady=(10, 0))
        self.textbox3 = customtkinter.CTkTextbox(self.sidebar_frame02, width=250, height=500)
        self.textbox3.grid(row=1, column=4, padx=(10, 0), pady=(20, 10), sticky="nsew")
        self.logo_label05_1 = customtkinter.CTkLabel(self.sidebar_frame02, text="Sure", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.logo_label05_1.grid(row=2, column=4, padx=10, pady=(10, 0))
        # set default values
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def agglom_bf_2(self):
        dialog = customtkinter.CTkInputDialog(text="Kullanici numarasi giriniz:", title="Agglo")
        user_id = dialog.get_input()  # waits for input
        user_id = int(user_id)
        # Kullanıcının ait olduğu küme
        user_cluster = features.loc[features['Friend 1'] == user_id, 'Cluster'].values[0]

        # Aynı kümedeki diğer kullanıcıları bul
        cluster_users = features.loc[
            features['Cluster'] == user_cluster, ['Friend 1', 'Name', 'Surname']].drop_duplicates()

        # Kullanıcının arkadaşlarını ve kümedeki diğer kullanıcıları göster
        recommendations = cluster_users[cluster_users['Friend 1'] != user_id].head(max_recommendations)

        self.textbox1.delete("0.0", "end")  # delete all text
        result_list = []
        # Öneri sonuçlarını liste şeklinde göster
        print(f"\nÖneriler (Küme {user_cluster}):")
        for index, row in recommendations.iterrows():
            print(f"{row['Friend 1']} - {row['Name']} {row['Surname']}")
            result_list.append(f"{row['Friend 1']} - {row['Name']} {row['Surname']}")
            aggloTextString = (f"{row['Friend 1']} - {row['Name']} {row['Surname']}")
            self.textbox1.insert("end", "\n" + result_list[-1])

    def knn2_bf_2(self):
        dialog = customtkinter.CTkInputDialog(text="Kullanici numarasi giriniz:", title="Knn2")
        user_id = dialog.get_input()  # waits for input
        user_id = int(user_id)
        # Kullanıcının yaşını al
        user_age = user_age_df_knn2.loc[user_id]

        # Kullanıcılar arasındaki uzaklıkları hesapla
        knn_model = NearestNeighbors(n_neighbors=3, metric='manhattan')
        flattened_features = user_age_df_knn2.values.reshape(-1, 1)
        knn_model.fit(flattened_features)

        distances, indices = knn_model.kneighbors([[user_age]])

        # Kullanıcının kümesindeki diğer kullanıcıları öneri olarak al
        cluster_users = [user for user in indices.flatten() if user != user_id]

        # Kullanıcıların isim ve soyisimlerini içeren bir DataFrame oluştur
        user_info_df = user_df_knn2.set_index('UserID')

        self.textbox2.delete("0.0", "end")  # delete all text
        result_list_knn2 = []
        # Öneri listesini yazdır
        if cluster_users:
            print(f"{user_id}. kullanıcının Arkadaşlık Önerileri:")
            for onerilen_kullanici in cluster_users:
                surname = user_info_df.loc[onerilen_kullanici, 'Surname']
                name = user_info_df.loc[onerilen_kullanici, 'Name']
                print(f"({name} {surname})")
                result_list_knn2.append(f"{name} {surname}")
                self.textbox2.insert("end", "\n" + result_list_knn2[-1])
        else:
            print(f"{user_id}. kullanıcının Arkadaşlık Önerisi Bulunamadı.")

    def kmeans_analy(self):
        st_kmeans = time.time()
        # Datasetleri oku
        friends_df = pd.read_csv(r'friends_table.csv')
        user_df = pd.read_csv(r'userID_table.csv')

        # Friend 1 ile UserID'yi birleştir
        merged_df = pd.merge(friends_df, user_df, left_on='Friend 1', right_on='UserID', how='inner')

        # Her bir kullanıcı için diğer kullanıcılarla olan ortak arkadaş sayılarını hesapla
        user_ids = merged_df['UserID'].unique()
        common_friends_matrix = []

        for user_id in user_ids:
            common_friends = [len(set(merged_df[merged_df['UserID'] == user_id]['Friend 2']).intersection(
                set(merged_df[merged_df['UserID'] == other_user]['Friend 2']))) for other_user in user_ids]
            common_friends_matrix.append(common_friends)

        # Ortak arkadaş sayılarını içeren bir DataFrame oluştur
        common_friends_df = pd.DataFrame(common_friends_matrix, index=user_ids, columns=user_ids)

        # K-Means kümeleme uygula
        kmeans = KMeans(n_clusters=3, random_state=42)
        common_friends_df['Cluster'] = kmeans.fit_predict(common_friends_df)

        # Kullanıcıdan giriş al
        # user_id = int(input("Arkadaşlık önerileri almak istediğiniz kullanıcının numarasını girin: "))

        dialog = customtkinter.CTkInputDialog(text="Kullanici numarasi giriniz:", title="Knn2")
        user_id = dialog.get_input()  # waits for input
        user_id = int(user_id)

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

        self.textbox3.delete("0.0", "end")  # delete all text
        result_list_kmeans = []
        # Öneri listesini yazdır
        if cluster_users:
            print(f"{user_id}. kullanıcının Arkadaşlık Önerileri:")
            for onerilen_kullanici in cluster_users:
                surname = user_info_df.loc[onerilen_kullanici, 'Surname']
                name = user_info_df.loc[onerilen_kullanici, 'Name']
                print(f"({name} {surname}) ile ortak arkadaş sayısı: {common_friends_df.loc[user_id, onerilen_kullanici]}")
                result_list_kmeans.append(f"{name} {surname}")
                self.textbox3.insert("end", "\n" + result_list_kmeans[-1])
        else:
            print(f"{user_id}. kullanıcının Arkadaşlık Önerisi Bulunamadı.")
        et_kmeans = time.time()
        elapsed_time_kmeans = (et_kmeans - st_kmeans) / 60
        print('Execution time:', elapsed_time_kmeans, 'minutes')
        elapsed_time_kmeans_str = str(elapsed_time_kmeans)
        app_kmeans = App()
        app_kmeans.logo_label05_1.configure(text="Sure" + elapsed_time_kmeans_str + "Dk")


def agglom_Analy():
    st_agglr = time.time()
    global islem_yapildi_mi
    islem_yapildi_mi = False
    global friends_data, user_data, merged_data
    global features, scaler, scaled_features, cluster_model
    global max_recommendations
    if not islem_yapildi_mi:
        # Verileri oku
        friends_data = pd.read_csv("friends_table.csv")
        user_data = pd.read_csv("userID_table.csv")
        reactions_data = pd.read_csv("reactions_table_m.csv")

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
        islem_yapildi_mi = True
        et_agglr = time.time()
        elapsed_time_knn = (et_agglr - st_agglr)/60
        print('Execution time:', elapsed_time_knn, 'minutes')
        elapsed_time_knn_str = str(elapsed_time_knn)
        app_agglo = App()
        app_agglo.logo_label03_1.configure(text="Sure" + elapsed_time_knn_str + "Dk")


def knn2_analy():
    st_knn2 = time.time()
    global friends_df_knn2, user_df_knn2, merged_df_knn2
    global user_ids_knn2, common_friends_matrix_knn2, common_friends_knn2, common_friends_df_knn2
    global user_age_df_knn2
    # Datasetleri oku
    friends_df_knn2 = pd.read_csv(r'friends_table.csv')
    user_df_knn2 = pd.read_csv(r'userID_table.csv')

    # Friend 1 ile UserID'yi birleştir
    merged_df_knn2 = pd.merge(friends_df_knn2, user_df_knn2, left_on='Friend 1', right_on='UserID', how='inner')

    # Her bir kullanıcı için diğer kullanıcılarla olan ortak arkadaş sayılarını hesapla
    user_ids_knn2 = merged_df_knn2['UserID'].unique()
    common_friends_matrix_knn2 = []

    for user_id_knn2 in user_ids_knn2:
        common_friends = [len(set(merged_df_knn2[merged_df_knn2['UserID'] == user_id_knn2]['Friend 2']).intersection(
            set(merged_df_knn2[merged_df_knn2['UserID'] == other_user]['Friend 2']))) for other_user in user_ids_knn2]
        common_friends_matrix_knn2.append(common_friends)

    # Ortak arkadaş sayılarını içeren bir DataFrame oluştur
    common_friends_df_knn2 = pd.DataFrame(common_friends_matrix_knn2, index=user_ids_knn2, columns=user_ids_knn2)

    # Kullanıcıların yaşlarını içeren bir DataFrame oluştur
    user_age_df_knn2 = user_df_knn2.set_index('UserID')['Age']

    et_knn2 = time.time()
    elapsed_time_agglo = (et_knn2 - st_knn2) / 60
    print('Execution time:', elapsed_time_agglo, 'minutes')
    elapsed_time_agglo_str = str(elapsed_time_agglo)
    app_knn = App()
    app_knn.logo_label04_1.configure(text="Sure" + elapsed_time_agglo_str + "Dk")


if __name__ == "__main__":
    knn2_analy()
    agglom_Analy()
    app = App()
    app.mainloop()
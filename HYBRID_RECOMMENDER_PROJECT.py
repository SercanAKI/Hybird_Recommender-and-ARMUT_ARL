
###################################
# PROJE: Hybrid Recommender System
###################################

# ID'si verilen kullanıcı için item-based ve user-based recomennder yöntemlerini kullanarak tahmin yapalım.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapalım.

#######################
 # Verinin Hazırlanması
#######################
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
movie = pd.read_csv('5.Hafta/Hybrid Recommender/movie.csv')
rating = pd.read_csv('5.Hafta/Hybrid Recommender/rating.csv')
movie.head()
rating.head()
# Adım 1: Movie ve Rating veri setlerini okutalım.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti

# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyelim.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.

df = movie.merge(rating, how="left", on="movieId")
df.iloc:[]
df.head()
df.shape

# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayalım.Toplam oy kullanılma sayısı 10000'un
# altında olan filmleri veri setinden çıkaralım.
# Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.

df["title"].value_counts().head()
comment_counts = pd.DataFrame(df["title"].value_counts())

# Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturalım.
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım.

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('5.Hafta/Hybrid Recommender/movie.csv')
    rating = pd.read_csv('5.Hafta/Hybrid Recommender/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()
user_movie_df.shape
user_movie_df.head()

##############################################################
# Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
##############################################################

# Adım 1: Rastgele bir kullanıcı id'si seçelim.
#random states=50 (kullanıcı ıd sı nı seçeriz)
#str olacağından ınt e çeviriyoruz
#sample ile örneklem seçeriz
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=50).values)
# 129261
# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturalım.
random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayalım.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)
##########################################################################
# Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
##########################################################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçelim ve
# movies_watched_df adında yeni bir dataframe oluşturalım.
#izlenen filmlere bakalım
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
len(movies_watched_df)
# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini
# taşıyan user_movie_count adında yeni bir dataframe oluşturalım.
# Ve yeni bir df oluşturuyoruz.
# sıfır olmayanları almak için notnull kullanıyoruz

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index() 


# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden users_same_movies adında bir liste oluşturalım.

user_movie_count.columns = ["userId", "movie_count"]
#len yazabiliriz boyut için
len(movies_watched) * 60/100

users_same_movies = user_movie_count[user_movie_count["movie_count"] > len(movies_watched) * 60/100]["userId"]
len(users_same_movies)

#####################################################################
# Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#####################################################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların
# id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyelim.

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturalım.
#her iki sütun arasındaki korelasyon katsayısını hesaplar.
final_df.T.corr()
#"unstack()" yöntemi, korelasyon matrisini, "stacked" halinden "unstacked" haline dönüştürür ve sonuç olarak, bir Seri oluşturur.
#"sort_values()" yöntemi, bu Seri'yi korelasyon katsayısına göre sıralar.
#"drop_duplicates()" yöntemi, aynı olan movıe ve benzeri değişken isimlerini çıkartır.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])

#sutunlara kullanıcıları almalıyız
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
corr_df[corr_df["user_id_1"] == random_user]

corr_df.head()


# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları
# filtreleyerek top_users adında yeni bir dataframe oluşturalım.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

#sıralayalım
top_users = top_users.sort_values(by='corr', ascending=False)

#user ıd 2 yı user ıd olarak değiştirelim
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
# Adım 4:  top_users dataframe’ine rating veri seti ile merge edelim.

rating = pd.read_csv('5.Hafta/Hybrid Recommender/rating.csv')

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings.head(50)
len(movies_watched)
#################################################################################
# Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#################################################################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturalım.
#bir noktadan bakmamak için hem corr(yüksek) hem rating(yüksek) üzerinden bakıyoruz
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2: Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini
# içeren recommendation_df adında yeni bir dataframe oluşturalım.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df[["movieId"]].nunique()

# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayalım.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydedelim.

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)
movies_to_be_recommend.head()
movies_to_be_recommend.shape

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getirelim.

movie.head()
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

############################
# Item-Based Recommendation
############################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapalım.


# Adım 1: movie,rating veri setlerini okutalım.
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
movie = pd.read_csv('5.Hafta/Hybrid Recommender/movie.csv')
rating = pd.read_csv('5.Hafta/Hybrid Recommender/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alalım.
user = 50485
user_rating = rating[(rating['userId'] == user)].copy() 
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"].iloc[1]
user_rating.sort_values('timestamp', inplace=True)
latest_movie = user_rating[user_rating["rating"] == 5]
latest_movie_id = latest_movie.sort_values('timestamp', ascending=False)["movieId"].iloc[1]
#latest_movie_id = 3929

user_movie_df.get("Bank Dick, The (1940)")
movie_name = movie[movie["movieId"] == latest_movie_id]["title"].values[0]
movie_name = user_movie_df[movie_name]
#movie_name=Bank Dick, The (1940)

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyelim.

movie_df = user_movie_df[movie[movie["movieId"] == latest_movie_id]["title"].values[0]]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayalım.

user_movie_df.corrwith(movie_name).sort_values(ascending=False)[1:6]


# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak verelim.

def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False)[1:10]

item_based_recommender("Pulp Fiction (1994)", user_movie_df)




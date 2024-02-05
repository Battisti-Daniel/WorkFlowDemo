import pandas as pd
import ast
import sklearn
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df_filmes = pd.read_csv("./dataset_filmes.csv")

df_elenco = pd.read_csv("./dataset_elenco.csv")


df_filmes = df_filmes.merge(df_elenco, on = "title")

#df_filmes.info()

df_filmes = df_filmes[["movie_id","title","overview","genres","keywords","cast","crew","runtime"]]

#df_filmes.shape

df_filmes.isnull().sum()

df_filmes.dropna(inplace = True)

df_filmes.isnull().sum()

df_filmes.duplicated().sum()

def converter(obj):
    return [i["name"]  for i in ast.literal_eval(obj)]

df_filmes["genres"] = df_filmes["genres"].apply(converter)

df_filmes["keywords"] = df_filmes["keywords"].apply(converter)

df_filmes["cast"] = df_filmes["cast"].apply(converter)

df_filmes["crew"] = df_filmes["crew"].apply(converter)


df_filmes["overview"] = df_filmes["overview"].apply(lambda x:x.split())

df_filmes["genres"] = df_filmes["genres"].apply(lambda x:[i.replace(" ", "") for i in x])

df_filmes["keywords"] = df_filmes["keywords"].apply(lambda x:[i.replace(" ", "") for i in x])

df_filmes["cast"] = df_filmes["cast"].apply(lambda x:[i.replace(" ", "") for i in x])

df_filmes["crew"] = df_filmes["crew"].apply(lambda x:[i.replace(" ", "") for i in x])

df_filmes["tags"] = df_filmes["overview"] + \
                    df_filmes["genres"] + \
                    df_filmes["keywords"] + \
                    df_filmes["cast"] + \
                    df_filmes["crew"] 

df_filmes_novo = df_filmes[["movie_id","title","tags"]]

df_filmes_novo["tags"] = df_filmes_novo["tags"].apply(lambda x:" ".join(x))

df_filmes_novo["tags"] = df_filmes_novo["tags"].apply(lambda x:x.lower())

parser_ps = PorterStemmer()

def stem(text):
    return " ".join([parser_ps.stem(i) for i in text.split()])

df_filmes_novo["tags"] = df_filmes_novo["tags"].apply(stem)

cv = CountVectorizer(max_features=7000,stop_words="english")

vectors = cv.fit_transform(df_filmes_novo["tags"]).toarray()

similaridades = cosine_similarity(vectors)

def sistema_recomendacao(filme):
    
    index = df_filmes_novo[df_filmes_novo["title"] == filme].index[0]
    
    distances = sorted(list(enumerate(similaridades[index])),reverse=True, key=lambda x:x[1])
    
    for i in distances[1:11]:
        print(df_filmes_novo.iloc[i[0]].title)

sistema_recomendacao("Pirates of the Caribbean: The Curse of the Black Pearl")
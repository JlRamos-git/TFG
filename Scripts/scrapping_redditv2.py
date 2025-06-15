# --- IMPORTACIONES ---
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from dateutil.relativedelta import relativedelta
from newspaper import Article, build
from pytrends.request import TrendReq
from wordcloud import WordCloud
import praw
import os
from datetime import datetime

# --- CONFIGURACIÓN DE REDDIT ---
reddit = praw.Reddit(
    client_id="zl-SzUdsqjNChR62cinbFw",
    client_secret="jfSua_HFXMM9jiNFlH8HOOZQN25EVQ",
    user_agent="TFG App by u/QueTFGdM ",
    username="QueTFGdM ",
    password="Javiloko00."
)

# --- CONFIGURACIONES INICIALES ---
nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(nltk.corpus.stopwords.words('english'))
economic_terms = ["economy", "inflation", "recession", "crisis", "tariff", "interest", "gdp", "unemployment"]  # ✅ Palabras clave económicas

# --- FUNCIONES DE PROCESAMIENTO ---

def clean_comment(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[@#]\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

def prepare_dataset(df):
    if df.empty:
        return df
    df = df[~df["body"].isin(["[deleted]", "[removed]"])]
    df["clean_text"] = df["body"].apply(clean_comment)
    df = df[~df["author"].str.contains("bot", case=False, na=False)]
    df["date"] = pd.to_datetime(df["created_utc"], unit='s').dt.date
    df["source"] = "reddit"
    df["title"] = None
    df["type"] = "comentario"
    df = df[["source", "date", "title", "clean_text", "type", "score"]]
    df = df[df["clean_text"].str.contains('|'.join(economic_terms), case=False, na=False)]  # ✅ Filtro temático aplicado
    return df

# --- FUNCIONES DE EXTRACCIÓN ---
def get_reddit_comments(query, start_date, end_date, subreddit=None, limit=100):
    comments = []
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    subreddit_obj = reddit.subreddit(subreddit) if subreddit else reddit.subreddit("all")

    try:
        for submission in subreddit_obj.search(query, sort="new", time_filter="all", syntax="lucene", limit=limit):
            if hasattr(submission, "created_utc"):
                created_time = int(submission.created_utc)
                if start_timestamp <= created_time <= end_timestamp:
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        if hasattr(comment, "body"):
                            comments.append({
                                "author": comment.author.name if comment.author else "[deleted]",
                                "body": comment.body,
                                "created_utc": int(comment.created_utc),
                                "score": comment.score
                            })
    except Exception as e:
        print(f"Error en subreddit {subreddit}: {e}")

    return pd.DataFrame(comments)

# --- FUNCIÓN PRINCIPAL ---
def build_full_reddit_dataset(query, subreddits, limit_per_month=100):
    all_data = []
    total_comments = 0
    log = []

    start_date = datetime.now() - relativedelta(years=10)
    end_date = datetime.now()

    fecha_actual = start_date

    while fecha_actual < end_date:
        fecha_siguiente = fecha_actual + relativedelta(months=1)
        print(f"Buscando desde {fecha_actual.strftime('%Y-%m-%d')} hasta {fecha_siguiente.strftime('%Y-%m-%d')}")

        comments_this_month = 0

        for sub in subreddits:
            df = get_reddit_comments(
                query,
                start_date=fecha_actual.strftime("%Y-%m-%d"),
                end_date=fecha_siguiente.strftime("%Y-%m-%d"),
                subreddit=sub,
                limit=limit_per_month
            )
            if df.empty:
                print(f"⚠️ Sin resultados para r/{sub} en ese mes, buscando sin filtro de keyword...")
                df = get_reddit_comments(
                    query="",
                    start_date=fecha_actual.strftime("%Y-%m-%d"),
                    end_date=fecha_siguiente.strftime("%Y-%m-%d"),
                    subreddit=sub,
                    limit=limit_per_month
                )
                if df.empty:
                    log.append({
                        "mes": fecha_actual.strftime("%Y-%m"),
                        "subreddit": sub,
                        "comentarios": 0
                    })

            if not df.empty:
                df["source_subreddit"] = sub
                all_data.append(df)
                comments_this_month += len(df)

        print(f"Comentarios descargados en este mes: {comments_this_month}")
        total_comments += comments_this_month

        fecha_actual = fecha_siguiente

    if all_data:
        df_final = pd.concat(all_data, ignore_index=True)
        df_final = prepare_dataset(df_final)
    else:
        df_final = pd.DataFrame()

    print(f"✅ Total de comentarios descargados en 10 años: {total_comments}")

    if log:
        df_log = pd.DataFrame(log)
        log_path = r"C:\Users\gonlo\Desktop\TFG\Datos\Raw\Reddit\reddit_comentarios_vacios.csv"
        df_log.to_csv(log_path, index=False)
        print(f"📝 Log guardado en: {log_path}")

    return df_final

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    subreddits = [
        "economics",
        "finance",
        "wallstreetbets",
        "investing",
        "personalfinance",
        "StockMarket",
        "CryptoCurrency",
        "business",
        "economy",
        "globaleconomy",
        "macroeconomics",
        "financialindependence",
        "Money",
        "worldnews"
    ]
    query = "economy"
    limit_per_month = 500

    df_reddit = build_full_reddit_dataset(query, subreddits, limit_per_month)

    print(df_reddit.head())

    if not df_reddit.empty:
        save_folder = r"C:\Users\gonlo\Desktop\TFG"
        os.makedirs(save_folder, exist_ok=True)
        today_date = datetime.now().strftime("%Y-%m-%d")
        file_name = f"reddit_comments_10years_{today_date}.csv"
        full_path = os.path.join(save_folder, file_name)
        df_reddit.to_csv(full_path, index=False)
        print(f"✅ Dataset guardado en: {full_path}")
else:
    print("❌ No se encontraron comentarios para el periodo especificado.")

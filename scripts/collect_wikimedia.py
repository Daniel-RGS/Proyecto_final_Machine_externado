from datetime import datetime, timedelta

import pandas as pd
import requests

from project_paths import DEFAULT_START_DATE, RAW_DIR, default_end_date, ensure_dir


PAGEVIEWS_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/user/{article}/daily/{start}/{end}"
)

ARTICLES = [
    "Iran-Israel_proxy_conflict",
    "Iran-Israel_relations",
    "Israel-Hamas_war",
    "Red_Sea_crisis",
]


def _fetch_article_pageviews(article, start, end):
    url = PAGEVIEWS_URL.format(article=article, start=start, end=end)
    headers = {"User-Agent": "osint-iran-conflict-ml1/1.0 (student project)"}
    response = requests.get(url, headers=headers, timeout=60)
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    rows = response.json().get("items", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"].str.slice(0, 8), format="%Y%m%d")
    return df[["date", "article", "views"]]


def collect_wikimedia_data(start_date=DEFAULT_START_DATE, end_date=None):
    """
    Descarga pageviews reales de Wikipedia como fuente publica de contexto/interes.
    """
    end_date = end_date or default_end_date()
    start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    end = (end_dt + timedelta(days=1)).strftime("%Y%m%d")

    frames = []
    for article in ARTICLES:
        print(f"Descargando Wikimedia pageviews: {article}")
        article_df = _fetch_article_pageviews(article, start, end)
        if not article_df.empty:
            frames.append(article_df)

    if not frames:
        print("No se recuperaron pageviews de Wikimedia.")
        return pd.DataFrame()

    raw_df = pd.concat(frames, ignore_index=True)
    daily_df = raw_df.groupby("date", as_index=False).agg(
        wiki_pageviews_total=("views", "sum"),
        wiki_articles_tracked=("article", "nunique"),
    )

    output_dir = ensure_dir(RAW_DIR / "wikimedia")
    raw_df.to_csv(output_dir / "wikimedia_pageviews_raw.csv", index=False)
    daily_df.to_csv(output_dir / "wikimedia_daily.csv", index=False)
    print(f"Datos Wikimedia guardados: {len(daily_df)} dias")
    return daily_df


if __name__ == "__main__":
    collect_wikimedia_data()

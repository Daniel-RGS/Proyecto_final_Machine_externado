from datetime import datetime

import pandas as pd
import requests

from project_paths import DEFAULT_START_DATE, RAW_DIR, default_end_date, ensure_dir


API_URL = "https://en.wikipedia.org/w/api.php"
ARTICLES = [
    "Iran-Israel proxy conflict",
    "Iran-Israel relations",
    "Israel-Hamas war",
    "Red Sea crisis",
]


def _fetch_revisions(article, start_date, end_date):
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": article,
        "rvprop": "timestamp|user",
        "rvlimit": "max",
        "rvdir": "newer",
        "rvstart": f"{start_date}T00:00:00Z",
        "rvend": f"{end_date}T23:59:59Z",
    }
    headers = {"User-Agent": "osint-iran-conflict-ml1/1.0 (student project)"}
    rows = []

    while True:
        response = requests.get(API_URL, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        for page in pages.values():
            for rev in page.get("revisions", []):
                rows.append(
                    {
                        "article": article,
                        "timestamp": rev["timestamp"],
                        "user": rev.get("user", ""),
                    }
                )
        if "continue" not in payload:
            break
        params.update(payload["continue"])

    return pd.DataFrame(rows)


def collect_wikipedia_revisions(start_date=DEFAULT_START_DATE, end_date=None):
    end_date = end_date or default_end_date()
    frames = []
    for article in ARTICLES:
        print(f"Descargando revisiones Wikipedia: {article}")
        df_article = _fetch_revisions(article, start_date, end_date)
        if not df_article.empty:
            frames.append(df_article)

    if not frames:
        print("No se recuperaron revisiones.")
        return pd.DataFrame()

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df["date"] = pd.to_datetime(raw_df["timestamp"]).dt.tz_localize(None).dt.date
    daily_df = raw_df.groupby("date", as_index=False).agg(
        wiki_revision_count=("timestamp", "count"),
        wiki_revision_users=("user", "nunique"),
        wiki_revision_articles=("article", "nunique"),
    )
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    output_dir = ensure_dir(RAW_DIR / "wikipedia_revisions")
    raw_df.to_csv(output_dir / "wikipedia_revisions_raw.csv", index=False)
    daily_df.to_csv(output_dir / "wikipedia_revisions_daily.csv", index=False)
    print(f"Revisiones guardadas: {len(raw_df)} registros, {len(daily_df)} dias")
    return daily_df


if __name__ == "__main__":
    collect_wikipedia_revisions()

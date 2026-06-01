from datetime import datetime
from email.utils import parsedate_to_datetime

import feedparser
import pandas as pd

from project_paths import RAW_DIR, ensure_dir


RSS_FEEDS = {
    "BBC_World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al_Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Google_News": (
        "https://news.google.com/rss/search?"
        "q=(Iran%20OR%20Tehran)%20(Israel%20OR%20Gaza%20OR%20Lebanon%20OR%20missile%20OR%20strike)"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
}

KEYWORDS = [
    "iran",
    "israel",
    "tehran",
    "tel aviv",
    "khamenei",
    "gaza",
    "lebanon",
    "middle east",
    "strike",
    "missile",
    "red sea",
    "houthi",
    "yemen",
    "hormuz",
    "iraq",
    "syria",
]


def is_relevant(title, description):
    text = (str(title) + " " + str(description)).lower()
    return any(keyword in text for keyword in KEYWORDS)


def collect_rss_data():
    """
    Recopila titulares recientes de feeds RSS para alimentar el dashboard y la fuente textual.
    """
    print("Recolectando feeds RSS...")
    all_entries = []

    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if not is_relevant(entry.title, entry.get("description", "")):
                    continue
                try:
                    published_at = parsedate_to_datetime(entry.published)
                    date_str = published_at.strftime("%Y-%m-%d")
                except Exception:
                    date_str = datetime.now().strftime("%Y-%m-%d")

                all_entries.append(
                    {
                        "source": source,
                        "date": date_str,
                        "title": entry.title,
                        "description": entry.get("description", ""),
                        "published": entry.get("published", ""),
                        "link": entry.link,
                    }
                )
        except Exception as exc:
            print(f"Error procesando feed {source}: {exc}")

    df = pd.DataFrame(all_entries)
    if df.empty:
        print("No se encontraron noticias relevantes en los feeds RSS actuales.")
        return df

    output_dir = ensure_dir(RAW_DIR / "rss")
    latest_path = output_dir / "rss_latest.csv"
    if latest_path.exists():
        existing_df = pd.read_csv(latest_path)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(subset=["link"])

    df = df.sort_values(["date", "source", "published", "title"]).reset_index(drop=True)
    df.to_csv(latest_path, index=False)
    print(f"[OK] {len(df)} titulares relevantes guardados en {latest_path}")

    daily_df = df.groupby(["date", "source"]).size().unstack(fill_value=0).reset_index()
    renamed_columns = {"date": "date"}
    for source in RSS_FEEDS:
        if source in daily_df.columns:
            renamed_columns[source] = f"rss_{source.lower()}_count"
    daily_df = daily_df.rename(columns=renamed_columns)

    count_columns = [column for column in daily_df.columns if column != "date"]
    daily_df["rss_total"] = daily_df[count_columns].sum(axis=1)

    features_path = output_dir / "rss_daily_features.csv"
    daily_df.to_csv(features_path, index=False)
    print(f"[OK] Features diarias de RSS actualizadas en {features_path}")
    return df


if __name__ == "__main__":
    collect_rss_data()

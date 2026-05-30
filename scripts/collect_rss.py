import feedparser
import pandas as pd
import os
from datetime import datetime
from email.utils import parsedate_to_datetime

RSS_FEEDS = {
    "BBC_World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Al_Jazeera": "https://www.aljazeera.com/xml/rss/all.xml"
}

KEYWORDS = ["iran", "israel", "tehran", "tel aviv", "khamenei", "gaza", "lebanon", "middle east", "strike", "missile"]

def is_relevant(title, description):
    text = (str(title) + " " + str(description)).lower()
    return any(kw in text for kw in KEYWORDS)

def collect_rss_data():
    """
    Recopila titulares recientes de feeds RSS.
    Nota: Los feeds RSS generalmente solo contienen noticias de las últimas horas/días.
    Para investigación histórica se usaría GDELT o BigQuery, pero esto sirve para el dashboard en vivo
    y para demostrar la capacidad técnica pedida en el proyecto.
    """
    print("Recolectando feeds RSS...")
    all_entries = []
    
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if is_relevant(entry.title, entry.get('description', '')):
                    try:
                        dt = parsedate_to_datetime(entry.published)
                        date_str = dt.strftime("%Y-%m-%d")
                    except:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        
                    all_entries.append({
                        "source": source,
                        "date": date_str,
                        "title": entry.title,
                        "link": entry.link
                    })
        except Exception as e:
            print(f"Error procesando feed {source}: {e}")
            
    df = pd.DataFrame(all_entries)
    
    if not df.empty:
        output_dir = os.path.join(os.path.dirname(__dirname__), "data", "raw", "rss")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "rss_latest.csv")
        
        # Si ya existe, combinamos y quitamos duplicados
        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            df = pd.concat([df_existing, df]).drop_duplicates(subset=['link'])
            
        df.to_csv(filepath, index=False)
        print(f"✅ {len(df)} titulares relevantes extraídos y guardados en {filepath}")
        
        # Para features de modelado: contar menciones por día y por fuente
        df_grouped = df.groupby(['date', 'source']).size().unstack(fill_value=0).reset_index()
        # Renombrar columnas
        col_names = {'date': 'date'}
        for src in RSS_FEEDS.keys():
            if src in df_grouped.columns:
                col_names[src] = f"rss_{src.lower()}_count"
        df_grouped = df_grouped.rename(columns=col_names)
        
        # Guardar para el modelo
        feat_path = os.path.join(output_dir, "rss_daily_features.csv")
        df_grouped.to_csv(feat_path, index=False)
        print(f"✅ Features diarias de RSS actualizadas en {feat_path}")
        
    else:
        print("No se encontraron noticias relevantes en los feeds RSS actuales.")

if __name__ == "__main__":
    collect_rss_data()

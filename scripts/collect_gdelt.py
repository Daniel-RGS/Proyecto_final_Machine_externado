import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_gdelt_volume(query, start_date, end_date):
    """
    Obtiene el volumen de noticias diario (TimelineVolRaw) de GDELT.
    """
    params = {
        "query": query,
        "mode": "TimelineVolRaw",
        "format": "json",
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S")
    }
    
    try:
        response = requests.get(GDELT_DOC_API, params=params)
        response.raise_for_status()
        data = response.json().get('timeline', [])
        
        if not data:
            return pd.DataFrame()
            
        # Extraer la serie de datos de la respuesta (la primera suele ser la relevante)
        series_data = data[0].get('data', [])
        df = pd.DataFrame(series_data)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            # Agrupar por día (la respuesta puede ser horaria)
            df['date_only'] = df['date'].dt.date
            df_daily = df.groupby('date_only')['value'].sum().reset_index()
            df_daily.columns = ['date', 'volume']
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            return df_daily
            
        return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        print(f"Error consultando GDELT Volume: {e}")
        return pd.DataFrame()

def fetch_gdelt_tone(query, start_date, end_date):
    """
    Obtiene el tono promedio diario (TimelineTone) de GDELT.
    """
    params = {
        "query": query,
        "mode": "TimelineTone",
        "format": "json",
        "startdatetime": start_date.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_date.strftime("%Y%m%d%H%M%S")
    }
    
    try:
        response = requests.get(GDELT_DOC_API, params=params)
        response.raise_for_status()
        data = response.json().get('timeline', [])
        
        if not data:
            return pd.DataFrame()
            
        series_data = data[0].get('data', [])
        df = pd.DataFrame(series_data)
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df['date_only'] = df['date'].dt.date
            # Promediar el tono por día
            df_daily = df.groupby('date_only')['value'].mean().reset_index()
            df_daily.columns = ['date', 'avg_tone']
            df_daily['date'] = pd.to_datetime(df_daily['date'])
            return df_daily
            
        return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        print(f"Error consultando GDELT Tone: {e}")
        return pd.DataFrame()

def collect_gdelt_data(start_date_str="2025-01-01", end_date_str="2026-05-30"):
    """
    Itera mes a mes (para evitar timeouts) recopilando volumen y tono.
    """
    print(f"Descargando datos de GDELT desde {start_date_str} hasta {end_date_str}...")
    query = '"Iran" AND "Israel"'
    
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    all_volume = []
    all_tone = []
    
    # Dividir en chunks de 30 días
    current = start_date
    pbar = tqdm(total=(end_date - start_date).days)
    
    while current < end_date:
        next_date = min(current + timedelta(days=30), end_date)
        
        vol_df = fetch_gdelt_volume(query, current, next_date)
        if not vol_df.empty:
            all_volume.append(vol_df)
            
        tone_df = fetch_gdelt_tone(query, current, next_date)
        if not tone_df.empty:
            all_tone.append(tone_df)
            
        pbar.update((next_date - current).days)
        current = next_date
        time.sleep(1) # Respetar rate limits
        
    pbar.close()
    
    # Consolidar
    if all_volume and all_tone:
        df_vol = pd.concat(all_volume).drop_duplicates(subset=['date'])
        df_tone = pd.concat(all_tone).drop_duplicates(subset=['date'])
        
        df_final = pd.merge(df_vol, df_tone, on='date', how='outer')
        
        # Guardar
        output_dir = os.path.join(os.path.dirname(__dirname__), "data", "raw", "gdelt")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "gdelt_daily.csv")
        df_final.to_csv(filepath, index=False)
        print(f"✅ Datos de GDELT guardados en {filepath}")
        return df_final
    else:
        print("❌ No se pudieron recuperar datos de GDELT.")
        return pd.DataFrame()

if __name__ == "__main__":
    collect_gdelt_data()

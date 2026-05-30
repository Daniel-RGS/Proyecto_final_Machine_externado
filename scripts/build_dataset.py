import os
import pandas as pd
import numpy as from datetime import datetime

def build_escalation_score(acled_df):
    """
    Construye el escalation_score (0-100) basado en los datos de ACLED.
    Fórmula propuesta:
    score = (eventos * 2) + (fatalidades * 0.5) + (explosiones * 3) + (batallas * 2.5)
    """
    if acled_df.empty:
        return pd.DataFrame(columns=['date', 'escalation_score'])
        
    acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])
    
    # Agrupar por fecha
    daily_stats = acled_df.groupby('event_date').agg(
        n_eventos=('event_id_cnty', 'count'),
        n_fatalidades=('fatalities', 'sum'),
        n_explosiones=('event_type', lambda x: (x == 'Explosions/Remote violence').sum()),
        n_batallas=('event_type', lambda x: (x == 'Battles').sum())
    ).reset_index()
    
    # Calcular el score base
    daily_stats['raw_score'] = (
        daily_stats['n_eventos'] * 2.0 +
        daily_stats['n_fatalidades'] * 0.5 +
        daily_stats['n_explosiones'] * 3.0 +
        daily_stats['n_batallas'] * 2.5
    )
    
    # Normalizar a 0-100
    max_score = daily_stats['raw_score'].max()
    min_score = daily_stats['raw_score'].min()
    
    if max_score > min_score:
        daily_stats['escalation_score'] = 100 * (daily_stats['raw_score'] - min_score) / (max_score - min_score)
    else:
        daily_stats['escalation_score'] = 0
        
    return daily_stats[['event_date', 'escalation_score']].rename(columns={'event_date': 'date'})

def build_dataset():
    """
    Integra las 3 fuentes de datos en un solo dataset diario.
    """
    print("Construyendo dataset integrado...")
    base_dir = os.path.dirname(__dirname__)
    raw_dir = os.path.join(base_dir, "data", "raw")
    
    # Rango de fechas base (Ene 2025 - Mayo 2026)
    date_rng = pd.date_range(start='2025-01-01', end='2026-05-30', freq='D')
    df_final = pd.DataFrame({'date': date_rng})
    
    # 1. Cargar y procesar ACLED (Target)
    acled_dir = os.path.join(raw_dir, "acled")
    acled_files = [f for f in os.listdir(acled_dir) if f.endswith('.csv')] if os.path.exists(acled_dir) else []
    
    if acled_files:
        acled_df = pd.read_csv(os.path.join(acled_dir, acled_files[0]))
        score_df = build_escalation_score(acled_df)
        df_final = pd.merge(df_final, score_df, on='date', how='left').fillna({'escalation_score': 0})
    else:
        print("⚠️ No hay datos de ACLED. El target será 0.")
        df_final['escalation_score'] = 0
        
    # 2. Cargar GDELT
    gdelt_file = os.path.join(raw_dir, "gdelt", "gdelt_daily.csv")
    if os.path.exists(gdelt_file):
        gdelt_df = pd.read_csv(gdelt_file)
        gdelt_df['date'] = pd.to_datetime(gdelt_df['date'])
        df_final = pd.merge(df_final, gdelt_df, on='date', how='left').fillna({'volume': 0, 'avg_tone': 0})
    else:
        print("⚠️ No hay datos de GDELT.")
        df_final['volume'] = 0
        df_final['avg_tone'] = 0
        
    # 3. Cargar RSS Features
    rss_file = os.path.join(raw_dir, "rss", "rss_daily_features.csv")
    if os.path.exists(rss_file):
        rss_df = pd.read_csv(rss_file)
        rss_df['date'] = pd.to_datetime(rss_df['date'])
        df_final = pd.merge(df_final, rss_df, on='date', how='left').fillna(0)
    else:
        print("⚠️ No hay features de RSS.")
        df_final['rss_total'] = 0
        
    # 4. Ingeniería de características (Lags y Temporales)
    df_final['day_of_week'] = df_final['date'].dt.dayofweek
    df_final['month'] = df_final['date'].dt.month
    df_final['days_since_start'] = (df_final['date'] - pd.to_datetime('2025-01-01')).dt.days
    
    # Lags del target
    df_final['score_lag1'] = df_final['escalation_score'].shift(1).fillna(0)
    df_final['score_lag7_avg'] = df_final['escalation_score'].rolling(window=7, min_periods=1).mean().shift(1).fillna(0)
    
    # Rellenar cualquier NaN restante
    df_final = df_final.fillna(0)
    
    # Guardar
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "dataset_diario.csv")
    df_final.to_csv(filepath, index=False)
    
    print(f"✅ Dataset final integrado guardado en {filepath}")
    print(f"Filas: {len(df_final)}, Columnas: {len(df_final.columns)}")
    return df_final

if __name__ == "__main__":
    build_dataset()

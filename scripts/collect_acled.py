import os
from datetime import datetime
import pandas as pd
import requests
from dotenv import load_dotenv
from project_paths import DEFAULT_START_DATE, RAW_DIR, default_end_date, ensure_dir

# Cargar variables de entorno
load_dotenv()

# ACLED API Endpoint
ACLED_URL = "https://api.acleddata.com/acled/read"
EMAIL = os.getenv("ACLED_EMAIL")
KEY = os.getenv("ACLED_KEY")

def collect_acled_data(start_date=DEFAULT_START_DATE, end_date=None):
    """
    Descarga eventos de conflicto de ACLED para Irán, Israel y EE.UU. en Medio Oriente.
    """
    end_date = end_date or default_end_date()
    if not EMAIL or not KEY:
        print("⚠️ Advertencia: No se encontraron las credenciales de ACLED (ACLED_EMAIL, ACLED_KEY) en .env")
        print("Guarde sus credenciales en un archivo .env en la raíz del proyecto para descargar datos reales.")
        return pd.DataFrame()

    print(f"Descargando datos de ACLED desde {start_date} hasta {end_date}...")
    
    params = {
        "email": EMAIL,
        "key": KEY,
        "event_date": f"{start_date}|{end_date}",
        "event_date_where": "BETWEEN",
        "country": "Iran|Israel|Lebanon|Syria|Yemen|Iraq", # Ampliado para capturar la escalada regional
        "country_where": "IN",
        "limit": 10000, # ACLED limita a veces
    }
    
    try:
        response = requests.get(ACLED_URL, params=params)
        response.raise_for_status()
        data = response.json().get('data', [])
        
        if not data:
            print("No se encontraron datos para los parámetros dados.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Guardar datos raw
        output_dir = ensure_dir(RAW_DIR / "acled")
        filename = f"acled_{start_date}_{end_date}.csv"
        filepath = output_dir / filename
        
        df.to_csv(filepath, index=False)
        print(f"✅ Descargados {len(df)} eventos. Guardados en {filepath}")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al consultar ACLED API: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df = collect_acled_data()
    if df.empty:
        print("La descarga falló o está usando credenciales vacías.")

import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

from project_paths import DEFAULT_START_DATE, RAW_DIR, default_end_date, ensure_dir


load_dotenv()

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAP_KEY = os.getenv("FIRMS_MAP_KEY")


def collect_firms_data(start_date=DEFAULT_START_DATE, end_date=None):
    """
    Descarga focos de calor reales de NASA FIRMS para una caja de Medio Oriente.

    Requiere FIRMS_MAP_KEY en .env. Si no hay llave, no genera sustitutos:
    el proyecto debe seguir sin esta fuente o documentar que queda pendiente.
    """
    end_date = end_date or default_end_date()
    if not MAP_KEY:
        print("No se encontro FIRMS_MAP_KEY en .env. No se generaran datos sinteticos.")
        return pd.DataFrame()

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    bbox = "34,25,60,40"  # lon_min,lat_min,lon_max,lat_max
    sensor = "VIIRS_SNPP_NRT"
    all_chunks = []

    current = start
    while current <= end:
        days = min(10, (end - current).days + 1)
        url = (
            f"{FIRMS_AREA_URL}/{MAP_KEY}/{sensor}/{bbox}/"
            f"{days}/{current.strftime('%Y-%m-%d')}"
        )
        print(f"Descargando FIRMS {current.date()} + {days} dias...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        if response.text.strip() and not response.text.lower().startswith("invalid"):
            from io import StringIO

            chunk = pd.read_csv(StringIO(response.text))
            if not chunk.empty:
                all_chunks.append(chunk)

        current = current + timedelta(days=days)

    if not all_chunks:
        print("NASA FIRMS no devolvio registros para el rango solicitado.")
        return pd.DataFrame()

    df = pd.concat(all_chunks, ignore_index=True).drop_duplicates()
    if "acq_date" in df.columns:
        df = df.rename(columns={"acq_date": "date"})

    output_dir = ensure_dir(RAW_DIR / "firms")
    output_path = output_dir / "firms_hotspots.csv"
    df.to_csv(output_path, index=False)
    print(f"Datos reales FIRMS guardados en {output_path}: {len(df)} registros")
    return df


if __name__ == "__main__":
    collect_firms_data()

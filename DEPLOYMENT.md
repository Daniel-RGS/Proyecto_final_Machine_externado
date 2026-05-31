# Despliegue del dashboard

## Streamlit Cloud

1. Subir este proyecto a GitHub con estos archivos incluidos:
   - `dashboard/app.py`
   - `dashboard/intelligence_engine.py`
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml`
   - `data/processed/dataset_diario.csv`
   - `data/raw/rss/rss_latest.csv`
   - `data/raw/wikimedia/wikimedia_pageviews_raw.csv`
   - `data/raw/wikipedia_revisions/wikipedia_revisions_raw.csv`
   - `models/best_escalation_model.pkl`
   - `models/model_metrics.json`
2. En Streamlit Cloud crear una app nueva desde el repositorio.
3. Usar como archivo principal:

```text
dashboard/app.py
```

4. No se requieren secretos para el dashboard actual. Las credenciales solo aplican si se actualizan fuentes opcionales como ACLED o NASA FIRMS.
5. Cuando termine el build, copiar la URL publica de Streamlit Cloud y ponerla en el README y en la entrega.

## Verificacion antes de publicar

Desde la raiz del proyecto:

```bash
python -m pip install -r requirements.txt
python scripts/check_dashboard.py
python -m streamlit run dashboard/app.py
```

La verificacion correcta debe mostrar `Dashboard check OK`.

## URL local

En macOS/Linux:

```bash
./run_dashboard_mac.sh
```

En Windows:

```powershell
.\run_dashboard.ps1
```

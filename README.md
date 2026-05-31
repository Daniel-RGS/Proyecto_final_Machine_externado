# OSINT - Escalada Iran-Israel-EE.UU.

Proyecto de monitoreo diario y análisis de presión pública alrededor del conflicto.

## Resumen rápido

- **Unidad de análisis:** registros diarios agregados por región/país (ver `data/processed/dataset_diario.csv`).
- **Variable objetivo (target):** indicador binario que marca si el día siguiente cae en el cuartil alto del `media_pressure_score` (score de presión mediática).
- **Fuentes integradas hoy:** Wikimedia Pageviews, Wikipedia Revisions, RSS (Google News, BBC, Al Jazeera).
- **Fuentes opcionales:** GDELT, ACLED (requiere credenciales), NASA FIRMS.

## Qué hay en este repositorio

- `dashboard/app.py`: aplicación Streamlit (UI, filtros, gráficos y pestañas de modelo y metodología).
- `scripts/`: utilidades para descargar fuentes, recomponer el dataset y refrescar el pipeline.
- `data/raw/`, `data/processed/`: descargas y dataset procesado (`dataset_diario.csv`).
- `models/`: artefactos y `model_metrics.json` con comparación de modelos.

## Modelado y métricas

El pipeline compara varios modelos supervisados y guarda el mejor según métricas temporales.

- Modelos probados: baseline dummy, `LogisticRegression`, `RidgeClassifier`, `KNN`, `RandomForest` (entre otros verificables en `models/model_metrics.json`).
- Métricas de evaluación principales: ROC-AUC, F1-score, Precision y Recall (ver `models/model_metrics.json`).

## Limitaciones metodológicas

- El target no es violencia verificada en terreno; es una proxy de presión mediática basada en fuentes abiertas.
- Los forecasts son a corto plazo y se construyen a partir del histórico del `media_pressure_score`.
- Algunos datos (ACLED, GDELT, FIRMS) pueden no estar presentes localmente o requerir credenciales.

## Ejecutar localmente (macOS/Linux)

1. Crear/activar entorno virtual (opcional) y instalar dependencias:

```bash
./venv/bin/pip install -r requirements.txt
```

2. Ejecutar Streamlit usando el `venv` incluido:

```bash
./venv/bin/streamlit run dashboard/app.py --server.port 8501
```

Si `8501` está ocupado, Streamlit propondrá un puerto alternativo en su salida.

## Ejecutar en Windows

Desde PowerShell:

```powershell
.\setup_windows.ps1
.\run_dashboard.ps1
```

## Actualizar datos y reentrenar

Para volver a descargar fuentes y reentrenar los modelos:

```bash
python scripts/refresh_pipeline.py
```

O en Windows:

```powershell
.\update_data.ps1
```

## Despliegue

Preparado para Streamlit Cloud o despliegues que apunten al fichero principal `dashboard/app.py`.
Ver `DEPLOYMENT.md` para pasos específicos de despliegue y variables de entorno.

## Archivos clave

- `data/processed/dataset_diario.csv` — dataset final usado por el dashboard.
- `data/processed/pipeline_status.json` — estado de la última corrida.
- `models/model_metrics.json` — comparación de métricas entre modelos.
- `models/best_escalation_model.pkl` — artefacto entrenado (si existe).

---

Si quieres, puedo añadir una sección breve con las instrucciones para publicar en Streamlit Cloud (deploy automático desde `main`) y completar `DEPLOYMENT.md`.

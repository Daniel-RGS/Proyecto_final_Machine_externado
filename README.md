# OSINT - Escalada Iran-Israel-EE.UU.

Sistema OSINT de monitoreo diario para modelar presion publica alrededor del conflicto
Iran-Israel-EE.UU. usando fuentes abiertas reales. La unidad de analisis es el dia.

## Que quedo ajustado para esta computadora

- El proyecto ya no depende del `venv/` que venia empaquetado desde macOS.
- Las rutas quedaron centralizadas en `scripts/project_paths.py`.
- Se agregaron scripts de Windows para instalar dependencias, actualizar datos y abrir el dashboard.
- El dashboard de Streamlit ahora muestra la probabilidad del modelo correctamente y reporta la ultima actualizacion del pipeline.

## Estructura principal

- `dashboard/app.py`: dashboard de Streamlit.
- `presentation/index.html`: presentacion HTML separada del dashboard.
- `scripts/refresh_pipeline.py`: actualiza fuentes, recompone dataset y reentrena modelos.
- `data/processed/dataset_diario.csv`: dataset integrado.
- `data/processed/pipeline_status.json`: estado de la ultima corrida.
- `models/best_escalation_model.pkl`: mejor modelo entrenado.
- `models/model_metrics.json`: comparacion de modelos.

## Instalacion en Windows

Desde PowerShell, parado en la raiz del proyecto:

```powershell
.\setup_windows.ps1
```

El script intenta usar primero el Python de runtime disponible en esta maquina:

`C:\Users\melo1\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

Si no existe, hace fallback a `python`.

## Actualizar datos y reentrenar

Actualizacion segura con fuentes base:

```powershell
.\update_data.ps1
```

Eso corre:

- Wikimedia Pageviews
- Wikipedia Revisions
- RSS / Google News / BBC / Al Jazeera
- construccion del dataset
- reentrenamiento y comparacion de modelos

Fuentes opcionales mas lentas o con credenciales:

```powershell
.\update_data.ps1 --include-gdelt
.\update_data.ps1 --include-acled
.\update_data.ps1 --include-firms
```

Notas operativas:

- `GDELT` puede demorar bastante o fallar por rate limit.
- `ACLED` requiere `ACLED_EMAIL` y `ACLED_KEY`.
- `NASA FIRMS` requiere `FIRMS_MAP_KEY`.
- El flujo por defecto deja esas fuentes fuera para no volver lento o fragil el servicio de Streamlit.

## Abrir el dashboard

En macOS/Linux:

```bash
./run_dashboard_mac.sh
```

Si el puerto `8501` ya esta ocupado, el script intenta abrir en el siguiente puerto libre.

En Windows:

```powershell
.\run_dashboard.ps1
```

URL local esperada:

- `http://localhost:8501`
- la URL que muestre Streamlit si `8501` esta ocupado

## Verificar antes de entregar o desplegar

```bash
python scripts/check_dashboard.py
```

La salida esperada es `Dashboard check OK`. Esta prueba ejecuta el dashboard con
`streamlit.testing`, valida que no haya excepciones y confirma que existan pestanas,
metricas, tablas y filtros interactivos.

## Despliegue publico

El proyecto esta listo para Streamlit Cloud. El archivo principal que se debe configurar
en la plataforma es:

```text
dashboard/app.py
```

La guia paso a paso esta en `DEPLOYMENT.md`. Despues de publicar, pega la URL publica
en este README y en la entrega final.

## Fuentes reales integradas hoy

- Wikimedia Pageviews
- Wikipedia Revisions
- RSS / Google News, BBC y Al Jazeera

Fuentes opcionales:

- GDELT DOC API
- ACLED
- NASA FIRMS

## Modelado

El pipeline compara varios modelos supervisados y guarda el mejor por F1 temporal:

- Dummy majority baseline
- Logistic Regression
- Ridge Classifier
- KNN Classifier
- Random Forest

## Archivos de salida

- `data/raw/`: descargas por fuente
- `data/processed/dataset_diario.csv`: dataset final
- `data/processed/pipeline_status.json`: resumen de la ultima corrida
- `models/model_metrics.json`: metricas comparadas
- `models/best_escalation_model.pkl`: artefacto final

## Observacion metodologica

El target actual no representa violencia confirmada en terreno. Representa si el dia
siguiente cae en el cuartil alto del score de presion publica observado en fuentes
abiertas. Esa decision mantiene el proyecto util aun cuando ACLED o FIRMS no esten
disponibles localmente.

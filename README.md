# OSINT - Escalada Irán-EE.UU.-Israel (2025-2026)

Este proyecto aplica Machine Learning a datos de fuentes abiertas (OSINT) para predecir el nivel de escalada en Medio Oriente.

## Requisitos

1. Python 3.10+
2. Crear entorno virtual e instalar dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Mac/Linux
   pip install -r requirements.txt
   ```

## Configuración

1. Renombra `.env.example` a `.env`
2. Edita `.env` y agrega tus credenciales de ACLED (email y API key).

## Uso

### 1. Recolección de Datos
Ejecuta los scripts en la carpeta `scripts/` para bajar datos reales:
```bash
python scripts/collect_acled.py
python scripts/collect_gdelt.py
python scripts/collect_rss.py
```

### 2. Construcción del Dataset
```bash
python scripts/build_dataset.py
```
Esto creará el dataset final en `data/processed/dataset_diario.csv`.

### 3. Dashboard
Para ver el panel interactivo:
```bash
streamlit run dashboard/app.py
```

### 4. Presentación HTML
Abre el archivo `presentation/index.html` en tu navegador web.

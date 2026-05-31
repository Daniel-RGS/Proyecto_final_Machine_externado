# Fuentes de datos y verificacion de requisitos

Este documento resume la procedencia de los datos, el diccionario minimo de variables
y la verificacion contra el enunciado del proyecto final de ML1.

## 1. Estado real de las bases de datos

La version actual del proyecto usa datos reales descargados desde APIs publicas. No se
generan eventos, conteos ni etiquetas con simulacion aleatoria.

### Fuentes reales integradas en `data/`

| Fuente | Tipo | Acceso | Archivos | Cobertura | Uso en el proyecto |
| --- | --- | --- | --- | --- | --- |
| Wikimedia Pageviews | Senal social / atencion publica | API publica REST de Wikimedia | `data/raw/wikimedia/wikimedia_pageviews_raw.csv`, `data/raw/wikimedia/wikimedia_daily.csv` | 2025-01-01 a 2026-05-30 | Fuente base para medir atencion publica diaria sobre articulos relacionados con Iran-Israel y conflictos regionales. |
| Wikipedia Revisions | Senal social / actividad editorial | MediaWiki API publica | `data/raw/wikipedia_revisions/wikipedia_revisions_raw.csv`, `data/raw/wikipedia_revisions/wikipedia_revisions_daily.csv` | 2025-01-01 a 2026-05-11 | Fuente base para medir actividad editorial diaria, cantidad de usuarios y articulos editados. |
| RSS / Google News, BBC y Al Jazeera | Fuente textual / noticias | RSS publico | `data/raw/rss/rss_latest.csv`, `data/raw/rss/rss_daily_features.csv` | 2026-05-16 a 2026-05-31 | Tercera fuente real. Aporta titulares y conteos diarios de noticias relevantes para el conflicto. |

### Fuentes adicionales intentadas u opcionales

| Fuente | Estado | Motivo | Script |
| --- | --- | --- | --- |
| GDELT DOC API | Opcional / intentada | Fuente gratuita de noticias, volumen y tono. En la verificacion del 2026-05-31 fallo primero por DNS del sandbox y luego la ejecucion con red completa avanzo demasiado lento; se detuvo sin generar archivos. | `scripts/collect_gdelt.py` |
| BBC RSS, Al Jazeera RSS y Google News RSS | Integrada como fuente textual reciente | En la verificacion del 2026-05-31 se obtuvieron 111 titulares relevantes: 99 de Google News, 10 de Al Jazeera y 2 de BBC. | `scripts/collect_rss.py` |
| ACLED | Opcional con credenciales | Requiere `ACLED_EMAIL` y `ACLED_KEY`. No se reemplaza con datos sinteticos si no hay acceso. | `scripts/collect_acled.py` |
| NASA FIRMS | Opcional con credenciales | Requiere `FIRMS_MAP_KEY`. No se reemplaza con datos sinteticos si no hay acceso. | `scripts/collect_firms.py` |

## 2. Dataset procesado

Archivo principal: `data/processed/dataset_diario.csv`

- Filas: 514
- Unidad de analisis: dia
- Cobertura: 2025-01-01 a 2026-05-29
- Target actual: `target_high_escalation_next_day`
- Fuente del target: `Wikimedia_public_attention_next_day_q75`
- Columnas: 43

El target clasifica si el dia siguiente cae en el cuartil alto del indice de presion
publica observada. Este target no representa violencia verificada en terreno; representa
un aumento relativo de atencion publica en fuentes abiertas.

## 3. Diccionario minimo de variables

| Variable | Descripcion | Procedencia |
| --- | --- | --- |
| `date` | Dia de observacion. | Integracion diaria del pipeline. |
| `wiki_pageviews_total` | Total diario de visitas a articulos Wikipedia monitoreados. | Wikimedia Pageviews. |
| `wiki_articles_tracked` | Numero de articulos con pageviews recuperados ese dia. | Wikimedia Pageviews. |
| `wiki_revision_count` | Numero diario de revisiones en articulos monitoreados. | Wikipedia Revisions. |
| `wiki_revision_users` | Numero de usuarios distintos que editaron articulos monitoreados. | Wikipedia Revisions. |
| `wiki_revision_articles` | Numero de articulos editados ese dia. | Wikipedia Revisions. |
| `volume` | Volumen noticioso diario si GDELT esta disponible; cero si no hay datos reales descargados. | GDELT opcional. |
| `avg_tone` | Tono promedio diario si GDELT esta disponible; cero si no hay datos reales descargados. | GDELT opcional. |
| `rss_al_jazeera_count` | Conteo diario de titulares relevantes de Al Jazeera RSS. | RSS textual. |
| `rss_bbc_world_count` | Conteo diario de titulares relevantes de BBC World RSS. | RSS textual. |
| `rss_google_news_count` | Conteo diario de titulares relevantes de Google News RSS. | RSS textual. |
| `rss_total` | Conteo diario total de titulares RSS relevantes. | RSS textual. |
| `firms_hotspots_count` | Conteo diario de hotspots FIRMS si se descarga la fuente con API key. | NASA FIRMS opcional. |
| `media_pressure_score` | Indice 0-100 de presion publica observado. | Feature engineering sobre Wikimedia/Wikipedia o GDELT si existe. |
| `target_next_day_score` | `media_pressure_score` desplazado al dia siguiente. | Construccion supervisada del pipeline. |
| `target_high_escalation_next_day` | Clase binaria: 1 si el dia siguiente queda en el cuartil alto del score. | Target de clasificacion. |
| `target_threshold_q75` | Umbral del percentil 75 usado para definir la clase positiva. | Dataset procesado. |
| `target_source` | Fuente metodologica usada para construir el target. | Dataset procesado. |

Las columnas con sufijo `_lag1` y `_7d_avg` son rezagos y promedios moviles calculados
solo con informacion historica para evitar fuga de informacion del futuro.

## 4. Verificacion contra el enunciado del proyecto

| Requisito del PDF | Estado actual | Evidencia / comentario |
| --- | --- | --- |
| Usar datos reales, gratuitos y publicos | Cumple | Wikimedia Pageviews, Wikipedia Revisions y RSS/Google News-BBC-Al Jazeera son fuentes publicas. No hay generacion aleatoria ni placeholders como datos. |
| Trabajar con entre 3 y 5 fuentes | Cumple | En `data/` hay 3 fuentes reales: Wikimedia Pageviews, Wikipedia Revisions y RSS/Google News-BBC-Al Jazeera. |
| Incluir al menos 1 fuente textual | Cumple | `data/raw/rss/rss_latest.csv` conserva titulares, descripciones, fecha de publicacion, fuente y enlace. |
| Incluir al menos 1 fuente estructurada u operativa | Cumple parcialmente | Pageviews, revisiones y conteos RSS son estructurados. No obstante, no son eventos operativos de conflicto como ACLED/UKMTO; integrar una fuente operativa fortaleceria la entrega. |
| Incluir 1 fuente adicional de contexto, movilidad o senal social | Cumple | Wikimedia Pageviews y Wikipedia Revisions funcionan como senales sociales/de atencion publica. |
| Definir unidad de analisis | Cumple | Unidad: dia. |
| Formular pregunta propia | Cumple | Prediccion de alta presion publica del dia siguiente alrededor del conflicto Iran-Israel-EE.UU. |
| Construir dataset integrado | Cumple | Existe `dataset_diario.csv` con señales de Wikimedia, Wikipedia Revisions y RSS. |
| Entrenar minimo 3 modelos | Cumple | Dummy baseline, Logistic Regression, Ridge Classifier, KNN Classifier y Random Forest. |
| Comparar con metricas adecuadas | Cumple | Accuracy, precision, recall, F1 y ROC AUC cuando aplica. |
| Dashboard web | Cumple localmente / pendiente de despliegue publico | Existe `dashboard/app.py`; el PDF pide URL accesible. |
| Documentar decisiones, sesgos y limitaciones | Cumple parcialmente | README, notebooks y dashboard lo explican; este documento centraliza la auditoria. |

## 5. Riesgo principal y recomendacion

El requisito minimo de 3 fuentes queda cubierto con las bases actuales. El riesgo que
permanece es cualitativo: ninguna de las fuentes integradas es una base operacional de
eventos de conflicto. Para fortalecer la justificacion, se recomienda intentar una cuarta
fuente real:

1. GDELT DOC API para `volume` y `avg_tone`, si la API responde sin rate limit ni demoras excesivas.
2. ACLED si se consiguen credenciales gratuitas academicas.
3. NASA FIRMS si se obtiene `FIRMS_MAP_KEY`.
4. UKMTO u otra fuente publica de incidentes mediante scraping responsable.

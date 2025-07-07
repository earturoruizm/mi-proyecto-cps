# -*- coding: utf-8 -*-
"""
Análisis Mejorado de Formalización de Contratos de Prestación de Servicios
Versión API Web para Google Cloud Run - COMPLETA
"""
# --- IMPORTS BÁSICOS Y DE TU SCRIPT ORIGINAL ---
import pandas as pd
import numpy as np
import io
import warnings
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import sys
import traceback
import zipfile

# --- IMPORTS PARA LA API WEB ---
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# ============================================================================
# CONFIGURACIÓN Y CARGA DE MODELOS (SE EJECUTA UNA VEZ AL INICIAR EL SERVICIO)
# ============================================================================
app = Flask(__name__)

warnings.filterwarnings("ignore", message=r"\[W007\]", category=UserWarning)
pd.options.display.float_format = '{:.2f}'.format

# CONSTANTES DE TU SCRIPT
SHEET_NAME = 'CPS'
COLUMNAS_NECESARIAS = ['Id Contrato', 'Id Contratista', 'Fecha inicio', 'Fecha fin', 'Dependencia', 'Valor mes', 'Origen recursos', 'Objeto', 'Nombres y apellidos contratista']
COL_ID_CONTRATO, COL_ID_CONTRATISTA, COL_FECHA_INICIO, COL_FECHA_FIN, COL_DEPENDENCIA, COL_VALOR_MES, COL_ORIGEN_RECURSOS, COL_OBJETO, COL_NOMBRE_CONTRATISTA = COLUMNAS_NECESARIAS
COL_DURACION_DIAS, COL_ANIO, COL_ID_CONTRATISTA_MASK, COL_INDICE_RECURRENCIA, COL_INDICE_SIMILITUD, COL_INDICE_FORMALIZACION, COL_NIVEL_FORMALIZACION, COL_CANT_CONTRATOS, COL_ANIOS_COMPARADOS_CANT, COL_ANIOS_COMPARADOS_LIST, COL_DIAS_CONTRATACION, COL_ID_CONTRATO_ULTIMO, COL_CLUSTER = (
    'Duración días', 'Año', 'Id Contratista Enmascarado', 'Índice de Recurrencia', 'Índice de Similitud', 'Índice de Formalización', 'Nivel de Formalización', 'Cantidad de Contratos', 'Cantidad de Años Comparados', 'Años Comparados (Lista)', 'Días Totales Contratado', 'Id Contrato Último Año', 'Cluster (Agrupación)')
MODELO_ST_NOMBRE = 'hiiamsid/sentence_similarity_spanish_es'
SPACY_MODEL_NAME = 'es_core_news_sm'
PERCENTIL_BAJO, PERCENTIL_MEDIO = 0.33, 0.66
NIVEL_BAJA, NIVEL_MEDIANA, NIVEL_ALTA, NIVEL_INDETERMINADO, NIVEL_UNICO_CONTRATO = 'Baja Formalización', 'Mediana Formalización', 'Alta Formalización', 'Indeterminado', 'Baja Formalización (Único Contrato)'

# CARGA DE MODELOS NLP
print("Iniciando carga de modelos de Lenguaje Natural (esto sucede solo una vez)...")
_models_loaded = False
_nlp_model = None
_st_model = None
_stopwords_es = []
try:
    _nlp_model = spacy.load(SPACY_MODEL_NAME)
    _stopwords_es = stopwords.words('spanish')
    print(f"Cargando modelo semántico: '{MODELO_ST_NOMBRE}'...")
    _st_model = SentenceTransformer(MODELO_ST_NOMBRE)
    _models_loaded = True
    print("✅ Modelos NLP cargados y listos.")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR MODELOS NLP: {e}")
    traceback.print_exc()

# ============================================================================
# BLOQUE COMPLETO DE FUNCIONES ORIGINALES (CON MODIFICACIÓN EN EXPORTAR_HTML)
# ============================================================================

def validar_excel_entrada(df: pd.DataFrame, columnas_requeridas: list) -> tuple[bool, list, list]:
    errores = []
    advertencias = []
    if df.empty:
        errores.append("El archivo Excel o la hoja 'CPS' están completamente vacíos.")
        return False, errores, advertencias
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if columnas_faltantes:
        errores.append(f"Faltan columnas esenciales: {', '.join(columnas_faltantes)}")
        return False, errores, advertencias
    if df[COL_ID_CONTRATISTA].isnull().all():
        errores.append(f"La columna '{COL_ID_CONTRATISTA}' está completamente vacía.")
        return False, errores, advertencias
    df_temp = df.copy()
    df_temp['inicio_dt'] = pd.to_datetime(df_temp[COL_FECHA_INICIO], errors='coerce')
    df_temp['fin_dt'] = pd.to_datetime(df_temp[COL_FECHA_FIN], errors='coerce')
    num_fechas_invalidas = (df_temp['inicio_dt'].isnull() | df_temp['fin_dt'].isnull()).sum()
    if num_fechas_invalidas > 0:
        advertencias.append(f"{num_fechas_invalidas} filas con fechas inválidas serán omitidas.")
    num_fechas_inconsistentes = (df_temp['fin_dt'] < df_temp['inicio_dt']).sum()
    if num_fechas_inconsistentes > 0:
        advertencias.append(f"{num_fechas_inconsistentes} filas con fecha fin anterior a fecha inicio serán omitidas.")
    num_problemas_valor = (pd.to_numeric(df[COL_VALOR_MES], errors='coerce').isnull() & df[COL_VALOR_MES].notnull()).sum()
    if num_problemas_valor > 0:
        advertencias.append(f"{num_problemas_valor} valores no numéricos en '{COL_VALOR_MES}' se tratarán como 0.")
    return not bool(errores), errores, advertencias

def preprocesar_texto(texto: str) -> str:
    if not isinstance(texto, str) or not texto.strip(): return ""
    if not _models_loaded or _nlp_model is None:
        global _stopwords_es
        if not _stopwords_es:
            try:
                _stopwords_es = stopwords.words('spanish')
            except Exception:
                _stopwords_es = []
        return " ".join(w for w in texto.lower().split() if w.isalpha() and w not in _stopwords_es)
    doc = _nlp_model(texto.lower())
    return ' '.join([token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in _stopwords_es])

def calcular_indice_recurrencia(grupo: pd.DataFrame) -> float:
    if grupo.empty: return 0.0
    total_dias_contratado = grupo[COL_DURACION_DIAS].sum()
    fecha_inicio_primero, fecha_fin_ultimo = grupo[COL_FECHA_INICIO].min(), grupo[COL_FECHA_FIN].max()
    if pd.isna(fecha_inicio_primero) or pd.isna(fecha_fin_ultimo) or fecha_fin_ultimo < fecha_inicio_primero: return 0.0
    total_dias_periodo = (fecha_fin_ultimo - fecha_inicio_primero).days + 1
    return min(max(total_dias_contratado / total_dias_periodo, 0.0), 1.0) if total_dias_periodo > 0 else 1.0

def calcular_indice_similitud_avanzado(grupo: pd.DataFrame) -> float:
    if not _models_loaded or _st_model is None: return 0.0
    objetos_validos = [preprocesar_texto(obj) for obj in grupo[COL_OBJETO].astype(str).tolist() if obj and isinstance(obj, str)]
    if len(objetos_validos) <= 1: return 0.0
    try:
        embeddings = _st_model.encode(objetos_validos, convert_to_tensor=True, show_progress_bar=False)
        sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()
        return float(np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)]))
    except Exception: return 0.0

def add_clusters(df, n_clusters=3):
    features = df[[COL_INDICE_RECURRENCIA, COL_INDICE_SIMILITUD]].dropna()
    if len(features) < n_clusters:
        df[COL_CLUSTER] = 'No Agrupado'
        return df
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[features.index, COL_CLUSTER] = kmeans.fit_predict(features).astype(str)
    df[COL_CLUSTER] = df[COL_CLUSTER].fillna('No Agrupado')
    cluster_map = {str(i): f"Grupo {i+1}" for i in range(n_clusters)}
    cluster_map['No Agrupado'] = 'No Agrupado'
    df[COL_CLUSTER] = df[COL_CLUSTER].map(cluster_map).fillna('No Agrupado')
    return df

def clasificar_formalizacion_percentil(row: pd.Series, q_bajo, q_medio) -> str:
    if row[COL_CANT_CONTRATOS] == 1: return NIVEL_UNICO_CONTRATO
    if pd.isna(row[COL_INDICE_FORMALIZACION]): return NIVEL_INDETERMINADO
    indice = row[COL_INDICE_FORMALIZACION]
    if q_bajo == q_medio and q_bajo > 0: return NIVEL_BAJA if indice < q_bajo else NIVEL_ALTA
    elif indice < q_bajo: return NIVEL_BAJA
    elif indice < q_medio: return NIVEL_MEDIANA
    else: return NIVEL_ALTA

def _apply_common_layout_enhancements(fig, title, height=600, width=None):
    layout_options = dict(
        title=dict(text=f"<b>{title}</b>", font=dict(size=18, color='#2c3e50'), x=0.5, xanchor='center'),
        plot_bgcolor='rgba(255,255,255,0.95)', paper_bgcolor='#f8f9fa',
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=11, color="#34495e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5, bgcolor='rgba(255,255,255,0.8)', bordercolor='#bdc3c7', borderwidth=1, font=dict(size=10)),
        margin=dict(l=70, r=40, t=80, b=70), height=height)
    if width is not None: layout_options['width'] = width
    fig.update_layout(**layout_options)
    fig.update_layout(showlegend=True, modebar_remove=['lasso2d', 'select2d'])
    return fig

def plot_heatmap_plotly_mejorado(df, index_col, columns_col, values_col, title, xaxis_title, yaxis_title, colorscale="Blues", format_annot=".0f", aggfunc='count', hover_name_value="Contratos"):
    try:
        df_pivot = df[[index_col, columns_col, values_col]].copy()
        df_pivot[index_col] = df_pivot[index_col].astype(str)
        df_pivot[columns_col] = pd.to_numeric(df_pivot[columns_col], errors='coerce').dropna().astype(int)
        if aggfunc == 'sum': df_pivot[values_col] = pd.to_numeric(df_pivot[values_col], errors='coerce').fillna(0)
        pivot_table = df_pivot.pivot_table(index=index_col, columns=columns_col, values=values_col, aggfunc=aggfunc).fillna(0).sort_index(axis=1)
        if pivot_table.empty: return go.Figure()
    except Exception as e:
        traceback.print_exc(); return go.Figure()
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values, x=pivot_table.columns.astype(str), y=pivot_table.index, colorscale=colorscale, text=pivot_table.values,
        texttemplate=f"%{{text:{format_annot}}}", textfont=dict(size=9, color='white'),
        hovertemplate=(f"<b>{yaxis_title}:</b> %{{y}}<br><b>{xaxis_title}:</b> %{{x}}<br><b>{hover_name_value}:</b> %{{z:{format_annot}}}<extra></extra>")))
    fig = _apply_common_layout_enhancements(fig, title, height=max(450, len(pivot_table.index) * 18 + 150))
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title, yaxis=dict(autorange='reversed'))
    return fig

def plot_tendencias_plotly(df, value_col, group_col, title, yaxis_title, aggfunc='sum', format_hover=',.0f'):
    df_agg = df.copy()
    df_agg[COL_ANIO] = pd.to_numeric(df[COL_ANIO], errors='coerce').dropna().astype(int)
    tendencia_df = df_agg.groupby([COL_ANIO, group_col])[value_col].agg(aggfunc).unstack().fillna(0).sort_index()
    if tendencia_df.empty: return go.Figure()
    fig = go.Figure()
    colors = px.colors.qualitative.Vivid
    for i, col in enumerate(tendencia_df.columns):
        fig.add_trace(go.Scatter(x=tendencia_df.index, y=tendencia_df[col], mode='lines+markers', name=str(col), line=dict(color=colors[i % len(colors)])))
    fig = _apply_common_layout_enhancements(fig, title, height=450)
    fig.update_layout(xaxis_title='Año', yaxis_title=yaxis_title, hovermode="x unified", xaxis=dict(dtick=1))
    return fig

def plot_sankey_plotly(df, source_col, target_col, value_col, title):
    sankey_data = df.groupby([source_col, target_col]).size().reset_index(name='Value')
    sankey_data.columns = ['Source', 'Target', 'Value']
    if sankey_data.empty: return go.Figure()
    all_nodes = list(pd.unique(sankey_data[['Source', 'Target']].values.ravel('K')))
    node_dict = {name: i for i, name in enumerate(all_nodes)}
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, label=all_nodes, color=px.colors.qualitative.Pastel),
        link=dict(source=sankey_data['Source'].map(node_dict), target=sankey_data['Target'].map(node_dict), value=sankey_data['Value']))])
    fig = _apply_common_layout_enhancements(fig, title, height=max(500, len(all_nodes) * 18 + 100))
    return fig

def plot_scatter_formalizacion_plotly(resultado_df, title):
    if resultado_df.empty: return go.Figure()
    colores_nivel = {NIVEL_BAJA: '#3498db', NIVEL_MEDIANA: '#f1c40f', NIVEL_ALTA: '#e74c3c', NIVEL_INDETERMINADO: '#95a5a6', NIVEL_UNICO_CONTRATO: '#2ecc71'}
    fig = px.scatter(resultado_df, x=COL_INDICE_RECURRENCIA, y=COL_INDICE_SIMILITUD, color=COL_NIVEL_FORMALIZACION, size=COL_CANT_CONTRATOS,
                     size_max=20, custom_data=[COL_ID_CONTRATISTA_MASK, COL_NOMBRE_CONTRATISTA, COL_DEPENDENCIA, COL_INDICE_FORMALIZACION, COL_VALOR_MES, COL_CANT_CONTRATOS, COL_DIAS_CONTRATACION, COL_CLUSTER],
                     color_discrete_map=colores_nivel)
    fig.update_traces(hovertemplate="<br>".join(["<b>ID:</b> %{customdata[0]}", "<b>Nombre:</b> %{customdata[1]}", "<b>Nivel:</b> %{marker.color}", "<b>Índice Formalización:</b> %{customdata[3]:.3f}", "<b>Núm. Contratos:</b> %{customdata[5]}"]))
    fig = _apply_common_layout_enhancements(fig, title)
    fig.update_layout(xaxis_title="Índice de Recurrencia", yaxis_title="Índice de Similitud", xaxis=dict(range=[-0.05, 1.05]), yaxis=dict(range=[-0.05, 1.05]))
    return fig

def plot_bar_formalizacion_por_dependencia(resultado_df, title):
    if resultado_df.empty: return go.Figure()
    count_df = resultado_df.groupby([COL_DEPENDENCIA, COL_NIVEL_FORMALIZACION]).size().unstack(fill_value=0)
    count_df['Total'] = count_df.sum(axis=1)
    count_df = count_df.sort_values(by='Total', ascending=True).drop(columns='Total')
    if count_df.empty: return go.Figure()
    colores_nivel = {NIVEL_BAJA: '#3498db', NIVEL_MEDIANA: '#f1c40f', NIVEL_ALTA: '#e74c3c', NIVEL_INDETERMINADO: '#95a5a6', NIVEL_UNICO_CONTRATO: '#2ecc71'}
    fig = px.bar(count_df, color_discrete_map=colores_nivel, barmode='stack', orientation='h', text_auto=True)
    fig = _apply_common_layout_enhancements(fig, title, height=max(400, len(count_df.index) * 18 + 150))
    fig.update_layout(xaxis_title='Número de Contratistas', yaxis_title='Dependencia')
    return fig

def plot_distribuciones_indices(resultado_df, title_prefix):
    if resultado_df.empty: return go.Figure()
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Índice Recurrencia', 'Índice Similitud', 'Índice Formalización'))
    indices_cols = [COL_INDICE_RECURRENCIA, COL_INDICE_SIMILITUD, COL_INDICE_FORMALIZACION]
    for i, col in enumerate(indices_cols):
        fig.add_trace(go.Histogram(x=resultado_df[col].dropna()), row=1, col=i+1)
    fig = _apply_common_layout_enhancements(fig, f"{title_prefix} - Distribución de Índices", height=400, width=1000)
    fig.update_layout(showlegend=False)
    return fig

def plot_recursos_formalizacion(resultado_df, df_original, title):
    if resultado_df.empty: return go.Figure()
    formalizable = resultado_df[resultado_df[COL_NIVEL_FORMALIZACION].isin([NIVEL_ALTA, NIVEL_MEDIANA])]
    if formalizable.empty: return go.Figure()
    recursos_df = formalizable.groupby([COL_DEPENDENCIA, COL_NIVEL_FORMALIZACION])[COL_VALOR_MES].sum().reset_index()
    recursos_df[COL_VALOR_MES] *= 12
    colores_nivel = {NIVEL_MEDIANA: '#f1c40f', NIVEL_ALTA: '#e74c3c'}
    fig = px.bar(recursos_df, x=COL_DEPENDENCIA, y=COL_VALOR_MES, color=COL_NIVEL_FORMALIZACION, color_discrete_map=colores_nivel, barmode='stack')
    fig = _apply_common_layout_enhancements(fig, title, height=600)
    fig.update_layout(xaxis_title="Dependencia", yaxis_title="Recursos Anuales Estimados (COP)")
    return fig

def exportar_informe_completo_html(resultado_df, df_original, figs, entidad, fecha_str, total_contratistas_analizados, anio_inicio, anio_fin, anio_ultimo_evaluado, count_alta, count_mediana, count_baja, count_unico):
    # ESTA FUNCIÓN HA SIDO MODIFICADA PARA DEVOLVER UN STRING HTML
    # Aquí va toda tu lógica para crear la larga cadena de texto `html_content`
    # Es la misma que tenías en tu script original, solo que al final devuelve el string.
    fecha_generacion = pd.Timestamp.now().strftime('%d de %B de %Y, %H:%M:%S')
    current_year = pd.Timestamp.now().year
    # ... (todo tu código para calcular los KPIs del resumen ejecutivo)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Informe de Formalización - {entidad}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto p-8">
            <h1 class="text-4xl font-bold text-center">Informe de Análisis de Formalización</h1>
            <h2 class="text-2xl text-center">{entidad}</h2>
    """
    for key, fig in figs.items():
        if fig and fig.data:
            html_content += f'<div class="bg-white shadow-lg rounded-lg my-8 p-4"><h2>{fig.layout.title.text}</h2>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>'
    
    html_content += """
        </div>
    </body>
    </html>
    """
    return html_content

# ============================================================================
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO (CON TU LÓGICA COMPLETA)
# ============================================================================
def procesar_archivo(file_bytes, nombre_entidad):
    start_time = time.time()
    print("-" * 60)
    print(f"🚀 Iniciando Análisis para la entidad: {nombre_entidad} 🚀")
    if not _models_loaded:
        raise RuntimeError("Los modelos NLP no se cargaron. El servicio no puede operar.")

    # --- 1. Carga y Validación ---
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=SHEET_NAME)
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo Excel. Asegúrese de que la hoja '{SHEET_NAME}' existe. Error: {e}")
    
    es_valido, errores_criticos, advertencias = validar_excel_entrada(df, COLUMNAS_NECESARIAS)
    if not es_valido:
        raise ValueError("Error de validación en el archivo: " + "; ".join(errores_criticos))
    if advertencias:
        print(f"Advertencias: {'; '.join(advertencias)}")

    # --- 2. Preprocesamiento ---
    print("🧹 Preprocesando y limpiando datos...")
    df_proc = df[COLUMNAS_NECESARIAS].copy()
    df_proc[COL_FECHA_INICIO] = pd.to_datetime(df_proc[COL_FECHA_INICIO], errors='coerce')
    df_proc[COL_FECHA_FIN] = pd.to_datetime(df_proc[COL_FECHA_FIN], errors='coerce')
    df_proc = df_proc.dropna(subset=[COL_FECHA_INICIO, COL_FECHA_FIN])
    df_proc = df_proc[df_proc[COL_FECHA_FIN] >= df_proc[COL_FECHA_INICIO]].copy()
    df_proc[COL_DURACION_DIAS] = (df_proc[COL_FECHA_FIN] - df_proc[COL_FECHA_INICIO]).dt.days + 1
    df_proc[COL_ANIO] = df_proc[COL_FECHA_INICIO].dt.year
    df_proc[COL_VALOR_MES] = pd.to_numeric(df_proc[COL_VALOR_MES], errors='coerce').fillna(0)
    df_original_proc = df_proc.copy()

    # --- 3. Generación de Gráficos Generales ---
    print("📊 Generando gráficos de visión general...")
    figs = {}
    figs['heatmap_cantidad'] = plot_heatmap_plotly_mejorado(df_original_proc, COL_DEPENDENCIA, COL_ANIO, COL_ID_CONTRATO, f"Distribución de Contratos por Dependencia y Año - {nombre_entidad}", "Año", "Dependencia")
    figs['heatmap_valor'] = plot_heatmap_plotly_mejorado(df_original_proc, COL_DEPENDENCIA, COL_ANIO, COL_VALOR_MES, f"Valor Mensual Contratado (COP) por Dependencia y Año - {nombre_entidad}", "Año", "Dependencia", format_annot=',.0f', aggfunc='sum')
    figs['tendencia_valor'] = plot_tendencias_plotly(df_original_proc, COL_VALOR_MES, COL_ORIGEN_RECURSOS, f"Tendencia Anual del Valor por Origen de Recursos - {nombre_entidad}", "Valor Mensual Total (COP)")
    figs['tendencia_cantidad'] = plot_tendencias_plotly(df_original_proc, COL_ID_CONTRATO, COL_ORIGEN_RECURSOS, f"Tendencia Anual de Contratos por Origen de Recursos - {nombre_entidad}", "Número de Contratos", aggfunc='count')
    figs['sankey_recursos'] = plot_sankey_plotly(df_original_proc, COL_ORIGEN_RECURSOS, COL_DEPENDENCIA, COL_ID_CONTRATO, f"Flujo de Contratos: Origen a Dependencia - {nombre_entidad}")
    
    # --- 4. Filtrado y Cálculo de Índices ---
    print("🧮 Calculando índices de formalización...")
    ultimo_anio = int(df_proc[COL_ANIO].max())
    anio_inicio = int(df_proc[COL_ANIO].min())
    contratistas_ultimo_anio = df_proc.loc[df_proc[COL_ANIO] == ultimo_anio, COL_ID_CONTRATISTA].unique()
    df_filtered = df_proc[df_proc[COL_ID_CONTRATISTA].isin(contratistas_ultimo_anio)].copy()
    
    unique_ids = df_filtered[COL_ID_CONTRATISTA].unique()
    masked_id_mapping = {id_val: f"CTO-{str(i+1).zfill(len(str(len(unique_ids)))+1)}" for i, id_val in enumerate(unique_ids)}
    df_filtered[COL_ID_CONTRATISTA_MASK] = df_filtered[COL_ID_CONTRATISTA].map(masked_id_mapping)

    grouped = df_filtered.groupby(COL_ID_CONTRATISTA)
    indices_list = [{'id': name, 'recurrencia': calcular_indice_recurrencia(group), 'similitud': calcular_indice_similitud_avanzado(group)} for name, group in grouped]
    indices_df = pd.DataFrame(indices_list).set_index('id')
    indices_df.columns = [COL_INDICE_RECURRENCIA, COL_INDICE_SIMILITUD]

    if len(indices_df) > 1:
        scaler = MinMaxScaler()
        indices_df[[COL_INDICE_RECURRENCIA, COL_INDICE_SIMILITUD]] = scaler.fit_transform(indices_df[[COL_INDICE_RECURRENCIA, COL_INDICE_SIMILITUD]])
    indices_df[COL_INDICE_FORMALIZACION] = indices_df[[COL_INDICE_RECURRENCIA, COL_INDICE_SIMILITUD]].mean(axis=1)

    # --- 5. Agregación y Clasificación ---
    print("📋 Agregando información y clasificando...")
    agregados = df_filtered.groupby(COL_ID_CONTRATISTA).agg(
        **{COL_DEPENDENCIA: pd.NamedAgg(column=COL_DEPENDENCIA, aggfunc=lambda x: x.mode()[0]),
           COL_NOMBRE_CONTRATISTA: pd.NamedAgg(column=COL_NOMBRE_CONTRATISTA, aggfunc='first'),
           COL_ID_CONTRATISTA_MASK: pd.NamedAgg(column=COL_ID_CONTRATISTA_MASK, aggfunc='first'),
           COL_CANT_CONTRATOS: pd.NamedAgg(column=COL_ID_CONTRATO, aggfunc='nunique'),
           COL_DIAS_CONTRATACION: pd.NamedAgg(column=COL_DURACION_DIAS, aggfunc='sum')})
    
    info_ultimo_contrato = df_filtered[df_filtered[COL_ANIO] == ultimo_anio].sort_values(by=COL_FECHA_FIN, ascending=False).groupby(COL_ID_CONTRATISTA).first()
    resultado_final = agregados.join(indices_df).join(info_ultimo_contrato[[COL_VALOR_MES]]).reset_index()

    resultado_final = add_clusters(resultado_final)
    
    contratistas_clasificables = resultado_final[resultado_final[COL_CANT_CONTRATOS] > 1].copy()
    q_bajo, q_medio = 0, 0
    if len(contratistas_clasificables) >= 3:
        q_bajo = contratistas_clasificables[COL_INDICE_FORMALIZACION].quantile(PERCENTIL_BAJO)
        q_medio = contratistas_clasificables[COL_INDICE_FORMALIZACION].quantile(PERCENTIL_MEDIO)
    
    resultado_final[COL_NIVEL_FORMALIZACION] = resultado_final.apply(lambda row: clasificar_formalizacion_percentil(row, q_bajo, q_medio), axis=1)
    
    conteo_niveles = resultado_final[COL_NIVEL_FORMALIZACION].value_counts()
    count_alta, count_mediana, count_baja, count_unico = conteo_niveles.get(NIVEL_ALTA, 0), conteo_niveles.get(NIVEL_MEDIANA, 0), conteo_niveles.get(NIVEL_BAJA, 0), conteo_niveles.get(NIVEL_UNICO_CONTRATO, 0)
    
    # --- 6. Generación de Gráficos Finales ---
    figs['scatter_formalizacion'] = plot_scatter_formalizacion_plotly(resultado_final, f"Mapeo de Contratistas por Índices de Formalización - {nombre_entidad}")
    figs['barras_dependencia'] = plot_bar_formalizacion_por_dependencia(resultado_final, f"Distribución de Niveles de Formalización por Dependencia - {nombre_entidad}")
    figs['distribucion_indices'] = plot_distribuciones_indices(resultado_final, nombre_entidad)
    figs['recursos_formalizacion'] = plot_recursos_formalizacion(resultado_final, df_original_proc, f"Recursos Anuales Estimados por Dependencia y Nivel - {nombre_entidad}")

    # --- 7. Exportación de Resultados EN MEMORIA ---
    print("💾 Empaquetando resultados en un archivo ZIP...")
    fecha_actual_str = pd.Timestamp.now().strftime('%Y%m%d')
    
    html_string_resultado = exportar_informe_completo_html(resultado_final, df_original_proc, figs, nombre_entidad, fecha_actual_str, len(resultado_final), anio_inicio, ultimo_anio, ultimo_anio, count_alta, count_mediana, count_baja, count_unico)

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        resultado_final.to_excel(writer, index=False, sheet_name='Analisis_Formalizacion')
    excel_buffer.seek(0)
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"Informe_Analisis_{secure_filename(nombre_entidad)}_{fecha_actual_str}.html", html_string_resultado)
        zf.writestr(f"Resultados_{secure_filename(nombre_entidad)}_{fecha_actual_str}.xlsx", excel_buffer.read())
    zip_buffer.seek(0)
    
    total_time = time.time() - start_time
    print(f"✅ Proceso para '{nombre_entidad}' completado en {total_time:.2f} segundos.")
    return zip_buffer

# ============================================================================
# ENDPOINTS DE LA API WEB (puntos de entrada)
# ============================================================================
@app.route("/analizar", methods=["POST"])
def analizar_endpoint():
    if 'archivo_excel' not in request.files:
        return jsonify({"error": "Petición inválida: falta el archivo 'archivo_excel'"}), 400
    file = request.files['archivo_excel']
    nombre_entidad = request.form.get("nombre_entidad", "Entidad_Desconocida")
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400
    if file and file.filename.endswith('.xlsx'):
        try:
            file_bytes = file.read()
            zip_con_resultados = procesar_archivo(file_bytes, nombre_entidad)
            return send_file(
                zip_con_resultados,
                as_attachment=True,
                download_name=f"Analisis_CPS_{secure_filename(nombre_entidad)}.zip",
                mimetype='application/zip'
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500
    return jsonify({"error": "Formato de archivo no válido. Se requiere .xlsx"}), 400

@app.route("/", methods=["GET"])
def index():
    return "<h1>Servicio de Análisis de Contratos CPS está activo.</h1>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
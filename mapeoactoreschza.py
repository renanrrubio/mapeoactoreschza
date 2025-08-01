import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from fpdf import FPDF
import base64
import tempfile
import os
from datetime import datetime
import plotly.io as pio
import networkx as nx
import numpy as np
import colorsys
# Asegúrate de que esta línea esté descomentada o añadida
from folium import plugins
import xml.etree.ElementTree as ET # Para generar KML
import folium
from streamlit_folium import st_folium
# Configuración de Plotly
pio.kaleido.scope.mathjax = None
# Configuración de la página
st.set_page_config(
    page_title="MAPEO DE ACTORES DE LOS VALLE CHANCAY - ZAÑA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- FUNCIONES DE UTILIDAD ---
# Función para resetear el estado del reporte
def reset_report_state():
    """Reinicia el estado del reporte en session_state."""
    if 'report_plots' in st.session_state:
        del st.session_state['report_plots']
    if 'report_stats' in st.session_state:
        del st.session_state['report_stats']
# Función auxiliar segura para convertir Hex a RGB
def hex_to_rgb_safe(hex_color):
    """Convierte un color hexadecimal a una tupla RGB (0-255). Maneja errores."""
    if not isinstance(hex_color, str):
        return (0, 123, 255) # Azul por defecto
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (0, 123, 255) # Azul por defecto
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 123, 255) # Azul por defecto
# --- FUNCIONES PARA PDF ---
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.set_auto_page_break(auto=True, margin=15)
        self.set_font('Arial', '', 10)
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Reporte Analítico - Excel Analytics Pro+', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')
    def add_section_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 10, title, 0, 1)
        self.ln(2)
    def add_plot(self, plot_path, caption="", width=180):
        if os.path.exists(plot_path):
            self.image(plot_path, x=(self.WIDTH - width)/2, w=width)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 5, caption, 0, 1, 'C')
            self.ln(5)
    def add_table(self, df, title=""):
        if title:
            self.add_section_title(title)
        col_width = self.WIDTH / len(df.columns) * 0.9
        row_height = self.font_size * 1.5
        # Encabezados
        self.set_fill_color(200, 220, 255)
        self.set_font('Arial', 'B', 9)
        for col in df.columns:
            self.cell(col_width, row_height, str(col), border=1, fill=True)
        self.ln(row_height)
        # Datos
        self.set_font('Arial', '', 8)
        for _, row in df.iterrows():
            for item in row:
                self.cell(col_width, row_height, str(item), border=1)
            self.ln(row_height)
        self.ln(5)
# --- FUNCIONES PARA KML ---
def generate_kml(data, lat_col, lon_col, name_col=None, description_cols=None, style_col=None, filename="mapa.kml"):
    """
    Genera un archivo KML a partir de coordenadas.
    """
    try:
        kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        document = ET.SubElement(kml, "Document")
        ET.SubElement(document, "name").text = filename.replace(".kml", "")
        styles = {}
        if style_col and style_col in data.columns:
            unique_styles = data[style_col].dropna().unique()
            color_palette = px.colors.qualitative.Set1
            for i, style_value in enumerate(unique_styles):
                style_id = f"style_{i}"
                styles[str(style_value)] = style_id
                style_element = ET.SubElement(document, "Style", id=style_id)
                icon_style = ET.SubElement(style_element, "IconStyle")
                hex_color = color_palette[i % len(color_palette)].lstrip('#')
                kml_color = f"ff{hex_color[4:6]}{hex_color[2:4]}{hex_color[0:2]}"
                ET.SubElement(icon_style, "color").text = kml_color
                ET.SubElement(icon_style, "scale").text = "1.0"
                icon = ET.SubElement(icon_style, "Icon")
                ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"
        for index, row in data.iterrows():
            placemark = ET.SubElement(document, "Placemark")
            name_text = str(row[name_col]) if name_col and name_col in data.columns and pd.notna(row[name_col]) else f"Ubicación {index+1}"
            ET.SubElement(placemark, "name").text = name_text
            if description_cols:
                desc_lines = [f"{col}: {row[col]}" for col in description_cols if col in data.columns and pd.notna(row[col])]
                if desc_lines:
                    desc_element = ET.SubElement(placemark, "description")
                    desc_element.text = "<![CDATA[" + "<br>".join(desc_lines) + "]]>"
            if style_col and style_col in data.columns and pd.notna(row[style_col]):
                style_key = str(row[style_col])
                if style_key in styles:
                     ET.SubElement(placemark, "styleUrl").text = f"#{styles[style_key]}"
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
                coordinates_text = f"{lon},{lat},0"
                point = ET.SubElement(placemark, "Point")
                ET.SubElement(point, "coordinates").text = coordinates_text
            except (ValueError, TypeError):
                 continue
        tree = ET.ElementTree(kml)
        kml_buffer = BytesIO()
        tree.write(kml_buffer, encoding='utf-8', xml_declaration=True)
        kml_buffer.seek(0)
        return kml_buffer.getvalue()
    except Exception as e:
        st.error(f"Error al generar KML: {e}")
        return None
# --- FUNCIONES PARA MAPA FOLIUM ---
# --- FUNCIONES PARA MAPA FOLIUM ---
# (Reemplaza tu función create_folium_map existente con esta)
def create_folium_map(data, lat_col, lon_col, name_col=None, popup_cols=None, style_col=None):
    """
    Crea un mapa interactivo usando Folium SIN clustering.
    Los marcadores se añaden directamente al mapa.
    """
    try:
        # --- 1. Validación y preparación de datos ---
        valid_coords_df = data.dropna(subset=[lat_col, lon_col])
        try:
            # Asegurarse de que las coordenadas sean numéricas
            valid_coords_df = valid_coords_df[pd.to_numeric(valid_coords_df[lat_col], errors='coerce').notnull() &
                                              pd.to_numeric(valid_coords_df[lon_col], errors='coerce').notnull()]
        except Exception:
            pass # Si falla, continuar con el DataFrame tal como está
        if valid_coords_df.empty:
            return None, "No hay datos válidos para mostrar en el mapa."
        # --- 2. Inicialización del mapa ---
        center_lat = valid_coords_df[lat_col].mean()
        center_lon = valid_coords_df[lon_col].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
        # --- 3. Configuración de colores basados en estilo (opcional) ---
        # Diccionario para almacenar colores por categoría de estilo
        style_colors = {}
        default_rgb = (0, 123, 255) # Color por defecto (azul)
        if style_col and style_col in valid_coords_df.columns:
            unique_styles = valid_coords_df[style_col].dropna().unique()
            if len(unique_styles) > 0:
                color_palette = px.colors.qualitative.Plotly # O usa Set1, Dark24, etc.
                for i, style_value in enumerate(unique_styles):
                    style_key = str(style_value)
                    hex_color = color_palette[i % len(color_palette)]
                    rgb_color = hex_to_rgb_safe(hex_color)
                    style_colors[style_key] = rgb_color
        # --- 4. Añadir marcadores directamente al mapa ---
        for _, row in valid_coords_df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (ValueError, TypeError):
                continue # Saltar filas con coordenadas no válidas
            # --- 5. Determinar el color del marcador ---
            marker_color = default_rgb
            icon_color_str = 'white' # Color por defecto del icono
            if style_col and style_col in valid_coords_df.columns and pd.notna(row[style_col]):
                style_val_key = str(row[style_col])
                if style_val_key in style_colors:
                    marker_color = style_colors[style_val_key]
                    # Ajustar color del ícono basado en el brillo del color del marcador
                    r, g, b = marker_color
                    brightness = (r * 299 + g * 587 + b * 114) / 1000
                    icon_color_str = 'white' if brightness < 128 else 'black'
            # --- 6. Crear contenido del popup y tooltip ---
            popup_content = ""
            if popup_cols:
                popup_lines = [f"<b>{col}:</b> {row[col]}" for col in popup_cols if col in valid_coords_df.columns and pd.notna(row[col])]
                popup_content = "<br>".join(popup_lines)
            else:
                # Si no hay popup_cols específicas, usar el nombre o una etiqueta genérica
                popup_name = str(row[name_col]) if name_col and name_col in valid_coords_df.columns and pd.notna(row[name_col]) else f"Punto ({lat:.4f}, {lon:.4f})"
                popup_content = f"<b>{popup_name}</b>"
            tooltip_text = str(row[name_col]) if name_col and name_col in valid_coords_df.columns and pd.notna(row[name_col]) else f"ID: {_}"
            # --- 7. Crear y añadir el marcador directamente al mapa (m) ---
            marker = folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=tooltip_text,
                icon=folium.Icon(color='lightgray', icon_color=icon_color_str, icon='map-marker', prefix='fa')
                # Nota: Folium tiene colores predefinidos limitados para 'color'.
                # Si deseas colores exactos por estilo, necesitas usar 'folium.CustomIcon' o CSS avanzado,
                # o simplemente usar el color para el icon_color y dejar el marcador gris.
                # Aquí usamos el color calculado para el icon_color y el marcador en gris claro.
            )
            marker.add_to(m) # <-- AQUÍ está el cambio clave: añadir directamente al mapa 'm'
        # --- 8. (Opcional) Añadir control de capas si se usa estilo
        # Como no hay clusters, no se añade LayerControl basado en clusters.
        # Si se usa 'style_col' para cambiar el icon_color, eso ya se refleja visualmente.
        return m, None
    except Exception as e:
        # Proporcionar más detalles del error puede ser útil
        return None, f"Error al crear el mapa: {str(e)}"
# --- FUNCIONES DE CARGA Y FILTRO DE DATOS ---
@st.cache_data(show_spinner="Cargando y procesando datos...", hash_funcs={BytesIO: lambda _: None})
def load_and_process(file, sheet_name=None):
    """Carga y valida el archivo Excel"""
    try:
        if not file:
            return None, "No se cargó ningún archivo"
        if file.size > 200 * 1024 * 1024:  # 200MB límite
            return None, "Archivo demasiado grande (límite: 200MB)"
        # Resetear estado del reporte al cargar nuevo archivo
        reset_report_state()
        engines = ['openpyxl', 'xlrd']
        for engine in engines:
            try:
                file.seek(0)
                excel_data = pd.ExcelFile(BytesIO(file.read()), engine=engine)
                df = pd.read_excel(excel_data, sheet_name=sheet_name) if sheet_name else pd.read_excel(excel_data)
                return df, "Datos cargados exitosamente"
            except Exception as e:
                continue
        return None, "No se pudo leer el archivo con ningún motor disponible"
    except Exception as e:
        return None, f"Error crítico: {str(e)}"
def setup_sidebar_filters(df):
    """Filtros interactivos con validación"""
    st.sidebar.header("🔍 Filtros Avanzados")
    available_cols = df.columns.tolist()
    selected_cols = st.sidebar.multiselect(
        "Columnas a mostrar",
        available_cols,
        default=available_cols[:min(10, len(available_cols))]
    )
    if not selected_cols:
        return None, "Selecciona al menos una columna"
    filtered_df = df[selected_cols]
    numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols:
        col_min, col_max = float(filtered_df[col].min()), float(filtered_df[col].max())
        user_range = st.sidebar.slider(
            f"Rango para {col}",
            col_min, col_max, (col_min, col_max),
            key=f"slider_{col}"
        )
        filtered_df = filtered_df[
            (filtered_df[col] >= user_range[0]) & 
            (filtered_df[col] <= user_range[1])
        ]
    categorical_cols = filtered_df.select_dtypes(exclude='number').columns.tolist()
    for col in categorical_cols:
        unique_vals = filtered_df[col].unique().tolist()
        selected_vals = st.sidebar.multiselect(
            f"Valores para {col}",
            unique_vals,
            default=unique_vals,
            key=f"multiselect_{col}"
        )
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
    return filtered_df, None
# --- FUNCIONES DE VISUALIZACIÓN ---
def create_optimized_radar(data, metrics, category=None):
    """Genera radar con SOLO LÍNEAS (y acepta ≥1 métrica)"""
    if not isinstance(data, pd.DataFrame) or data.empty:
        return None, "Datos no válidos"
    if len(metrics) < 1:
        return None, "Se necesita al menos 1 métrica"
    valid_metrics = [m for m in metrics if m in data.select_dtypes(include='number').columns]
    if len(valid_metrics) < 1:
        return None, "Métricas deben ser numéricas"
    try:
        fig = go.Figure()
        if category and category in data.columns:
            grouped = data.groupby(category)[valid_metrics].mean().reset_index()
            for metric in valid_metrics:
                fig.add_trace(go.Scatterpolar(
                    r=grouped[metric].tolist() + [grouped[metric].iloc[0]],
                    theta=grouped[category].tolist() + [grouped[category].iloc[0]],
                    name=metric,
                    line=dict(color=px.colors.qualitative.Plotly[len(fig.data)], width=2.5),
                    mode='lines+markers',
                    marker=dict(size=8, opacity=0.8),
                    hoverinfo='r+name'
                ))
            title = f"{category}"
        else:
            means = data[valid_metrics].mean()
            for metric in valid_metrics:
                fig.add_trace(go.Scatterpolar(
                    r=[means[metric]] * (len(valid_metrics)+1),
                    theta=valid_metrics + [valid_metrics[0]],
                    name=metric,
                    line=dict(color=px.colors.qualitative.Plotly[len(fig.data)], width=2.5),
                    mode='lines+markers',
                    marker=dict(size=8)
                ))
            title = "Radar de Métricas (Promedio)"
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[1, data[valid_metrics].max().max() * 1.1],
                    gridcolor='rgba(200, 200, 200, 0.5)',
                    linecolor='gray'
                ),
                angularaxis=dict(
                    rotation=90,
                    direction='clockwise',
                    linecolor='gray'
                )
            ),
            title=dict(text=title, x=0.5, font=dict(size=18)),
            legend=dict(
                orientation="h",  # Horizontal
                yanchor="top",   # Anclaje en la parte superior del área de la leyenda
                y=-0.15,         # Posición Y ligeramente por debajo del gráfico (ajusta si es necesario)
                xanchor="center", # Anclaje en el centro horizontal
                x=0.5            # Centrado horizontalmente
            ),
            height=550,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='closest'
        )
        return fig, None
    except Exception as e:
        return None, f"Error al generar radar: {str(e)}"
# --- FUNCIONES DE VISUALIZACIÓN ---
# (Mantener las otras funciones como create_optimized_radar, create_sociogram, etc., igual)
def create_power_interest_matrix(data, power_col, interest_col, stakeholder_col):
    """Genera matriz de Poder vs. Interés con cuadrantes mejorados, resultados y manejo de superposición."""
    try:
        required_cols = [power_col, interest_col, stakeholder_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return None, f"Columnas faltantes: {', '.join(missing_cols)}"
        # --- 1. Preparación de datos ---
        # Trabajar con una copia para no modificar el original
        plot_data = data.copy()
        # Asegurarse de que las columnas de interés y poder sean numéricas
        plot_data[interest_col] = pd.to_numeric(plot_data[interest_col], errors='coerce')
        plot_data[power_col] = pd.to_numeric(plot_data[power_col], errors='coerce')
        # Eliminar filas con valores NaN en las columnas críticas
        plot_data = plot_data.dropna(subset=[interest_col, power_col, stakeholder_col])
        if plot_data.empty:
             return None, "No hay datos válidos para mostrar en la matriz después de la limpieza."
        # --- 2. Cálculo de límites y puntos medios (MODIFICADO) ---
        # Establecer límites fijos del 1 al 5 con un pequeño padding para visualización
        fixed_min = 1
        fixed_max = 5
        range_padding = 0.05 # 5% de padding

        # Calcular el padding absoluto basado en el rango fijo
        padding_value = (fixed_max - fixed_min) * range_padding

        # Establecer los rangos fijos con padding
        y_range = [fixed_min - padding_value, fixed_max + padding_value]
        x_range = [fixed_min - padding_value, fixed_max + padding_value]

        # Calcular los puntos medios basados en el rango fijo
        power_mid = (fixed_max + fixed_min) / 2
        interest_mid = (fixed_max + fixed_min) / 2
        # --- Fin de la sección modificada ---
        
        # --- 3. Asignación de Cuadrantes (usando datos originales limpios) ---
        plot_data['cuadrante'] = plot_data.apply(
            lambda row: (
                "Gestionar Activamente" if (row[power_col] >= power_mid and row[interest_col] >= interest_mid) else
                "Mantener informados" if (row[power_col] < power_mid and row[interest_col] >= interest_mid) else
                "Mantener Satisfechos" if (row[power_col] >= power_mid and row[interest_col] < interest_mid) else
                "Monitorear"
            ), axis=1
        )
        # --- 4. Manejo de Superposición: Agrupar y Agregar Contador ---
        # Redondear coordenadas para agrupar puntos cercanos/iguales
        DECIMALS_FOR_GROUPING = 5 # Ajusta según la precisión necesaria
        plot_data['_group_lat'] = plot_data[power_col].round(DECIMALS_FOR_GROUPING)
        plot_data['_group_lon'] = plot_data[interest_col].round(DECIMALS_FOR_GROUPING)
        # Agrupar por coordenadas redondeadas y cuadrante
        grouped_data = plot_data.groupby(['_group_lat', '_group_lon', 'cuadrante'], as_index=False).agg({
            stakeholder_col: lambda x: list(x), # Lista de stakeholders
            power_col: 'first',    # Valor de poder (es el mismo para el grupo)
            interest_col: 'first', # Valor de interés (es el mismo para el grupo)
            'cuadrante': 'first'   # Cuadrante (es el mismo para el grupo)
        })
        # Renombrar la columna de lista para claridad
        grouped_data.rename(columns={stakeholder_col: 'stakeholders_list'}, inplace=True)
        # Calcular el conteo de puntos por grupo
        grouped_data['count'] = grouped_data['stakeholders_list'].apply(len)
        # Eliminar columnas temporales
        plot_data.drop(columns=['_group_lat', '_group_lon'], inplace=True, errors='ignore')
        # --- 5. Creación del Gráfico con Datos Agrupados ---
        fig = go.Figure()
        # Definir colores para cuadrantes
        quadrant_colors = {
            "Gestionar Activamente": '#2ca02c', # Verde
            "Mantener informados": '#1f77b4',           # Azul
            "Mantener Satisfechos": '#ff7f0e', # Naranja
            "Monitorear": '#d62728'   # Rojo
        }
        # Contar elementos por cuadrante para incluir en la leyenda
        total_counts = plot_data['cuadrante'].value_counts().to_dict()
        # Iterar por cada cuadrante
        for cuadrante, color in quadrant_colors.items():
            df_cuadrante_grouped = grouped_data[grouped_data['cuadrante'] == cuadrante]
            count_total = total_counts.get(cuadrante, 0) # Total de puntos en este cuadrante
            if not df_cuadrante_grouped.empty:
                # Incluir el conteo total en el nombre de la serie para la leyenda
                legend_name = f"{cuadrante} ({count_total} elementos)"
                # Crear texto para el hover que incluye todos los stakeholders del grupo
                hover_texts = []
                for _, row in df_cuadrante_grouped.iterrows():
                    if row['count'] == 1:
                         # Si solo hay uno, mostrar su información directamente
                        stakeholder_name = row['stakeholders_list'][0]
                        hover_text = (f"<b>{stakeholder_name}</b><br>"
                                      f"Interés: {row[interest_col]:.{DECIMALS_FOR_GROUPING}f}<br>"
                                      f"Poder: {row[power_col]:.{DECIMALS_FOR_GROUPING}f}<br>"
                                      f"Cuadrante: {row['cuadrante']}")
                    else:
                        # Si hay varios, mostrar el conteo y una lista
                        stakeholders_str = "<br>".join([f"• {name}" for name in row['stakeholders_list']])
                        hover_text = (f"<b>{row['count']} Stakeholders en este punto:</b><br>"
                                      f"{stakeholders_str}<br>"
                                      f"<b>Coordenadas:</b> ({row[interest_col]:.{DECIMALS_FOR_GROUPING}f}, {row[power_col]:.{DECIMALS_FOR_GROUPING}f})<br>"
                                      f"<b>Cuadrante:</b> {row['cuadrante']}")
                    hover_texts.append(hover_text)
                fig.add_trace(go.Scatter(
                    x=df_cuadrante_grouped[interest_col], # Eje X: Interés
                    y=df_cuadrante_grouped[power_col],    # Eje Y: Poder
                    mode='markers+text', # Mostrar marcadores y texto
                    name=legend_name,
                    text=df_cuadrante_grouped['count'], # Texto del marcador: el conteo
                    textposition="middle center", # Posición del texto
                    textfont=dict(size=10, color='white'), # Fuente del texto
                    marker=dict(
                        size=df_cuadrante_grouped['count'] * 3 + 10, # Tamaño basado en el conteo (ajustable)
                        color=color,
                        line=dict(width=1, color='DarkSlateGrey'),
                        sizemode='diameter' # Modo de tamaño
                    ),
                    hoverinfo='text',
                    hovertext=hover_texts # Texto personalizado para el hover
                ))
        # --- 6. Líneas divisorias y Rectángulos de fondo ---
        # Líneas divisorias
        fig.add_shape(type="line",
                     x0=interest_mid, y0=y_range[0], x1=interest_mid, y1=y_range[1],
                     line=dict(color="gray", width=1.5, dash="dot"))
        fig.add_shape(type="line",
                     x0=x_range[0], y0=power_mid, x1=x_range[1], y1=power_mid,
                     line=dict(color="gray", width=1.5, dash="dot"))
        # Rectángulos de fondo (usando los rangos calculados)
        fig.add_shape(type="rect",
            x0=x_range[0], y0=power_mid, x1=interest_mid, y1=y_range[1], # Mantener Satisfechos
            fillcolor="rgba(255, 228, 181, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect",
            x0=interest_mid, y0=power_mid, x1=x_range[1], y1=y_range[1], # Gestionar Activamente
            fillcolor="rgba(144, 238, 144, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect",
            x0=x_range[0], y0=y_range[0], x1=interest_mid, y1=power_mid, # Mantener Informados
            fillcolor="rgba(255, 182, 193, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect",
            x0=interest_mid, y0=y_range[0], x1=x_range[1], y1=power_mid, # Monitorear
            fillcolor="rgba(173, 216, 230, 0.1)", line=dict(width=0))
        # --- 7. Layout del Gráfico ---
        fig.update_layout(
            title="Matriz de Interés vs. Poder (Puntos agrupados con contador)",
            xaxis_title="Interés",
            yaxis_title="Poder",
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            height=600,
            margin=dict(l=50, r=50, t=80, b=100), # Más espacio abajo para la leyenda
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,240,240,0.8)',
             # Etiquetas en los cuadrantes (puedes mantenerlas o eliminarlas si la leyenda es suficiente)
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3, # Ajustado para dar espacio
                xanchor="center",
                x=0.5,
                traceorder='normal',
                font=dict(size=10) # Tamaño de fuente más pequeño para la leyenda
            )
        )
        # Preparar datos de resultados usando el DataFrame original sin agrupar para clasificación precisa
        result_df = plot_data[['cuadrante', stakeholder_col, interest_col, power_col]].sort_values(
            by=['cuadrante', interest_col, power_col], ascending=[True, False, False]
        )
        return fig, result_df
    except Exception as e:
        import traceback
        st.error(f"Error detallado en la matriz: {e}")
        # st.code(traceback.format_exc()) # Para depuración, opcional
        return None, f"Error al generar matriz: {str(e)}"
# --- FUNCIONES DE VISUALIZACIÓN ---
def create_sociogram(data, source_col, target1_col, target2_col=None, weight_col=None, node_size_col=None, layout='circular', include_target1=True, include_target2=True):
    """
    Crea un sociograma (diagrama de red) a partir de datos de relaciones.
    Ahora soporta una o dos columnas de destino, muestra una leyenda y permite incluir/excluir tipos de relaciones.
    Todos los nodos de Origen se dibujan, incluso si no tienen relaciones.
    Nodos Origen: Amarillo. Nodos Colaboraciones: Verde. Nodos Conflictos: Rojo. Aristas: Negras.
    Leyenda: Origen, Colaboraciones, Conflictos.
    """
    try:
        # --- 1. Validación de entrada ---
        if not isinstance(data, pd.DataFrame):
            return None, "Los datos deben ser un DataFrame de pandas"
        if data.empty:
            return None, "El conjunto de datos está vacío"

        # Validación de columnas basada en inclusión
        required_cols = [source_col]
        if include_target1 and target1_col:
            required_cols.append(target1_col)
        if include_target2 and target2_col:
            required_cols.append(target2_col)

        # target2_col es opcional, pero si se proporciona, debe existir
        if target2_col and target2_col not in data.columns:
             return None, f"La columna de destino 2 '{target2_col}' no existe en los datos."

        # Filtrar filas que tengan al menos un origen válido
        clean_data_for_relations = data.dropna(subset=[source_col])
        if clean_data_for_relations.empty:
            return None, "No hay datos válidos después de eliminar filas sin Origen"

        # --- 2. Creación del grafo (NetworkX) ---
        G = nx.DiGraph()
        # Identificar todos los nodos de origen únicos desde el dataset completo (filtrado)
        all_source_nodes_from_data = set(data[source_col].dropna().astype(str))

        # Añadir todos los nodos de origen al grafo desde el principio
        for source_node in all_source_nodes_from_data:
             G.add_node(source_node)

        # Sets para rastrear roles
        source_nodes_in_graph = set(all_source_nodes_from_data) # Todos los orígenes están en el grafo
        target1_nodes = set()
        target2_nodes = set() if target2_col else set()

        # --- 3. Procesar relaciones ---
        # Solo procesar relaciones si están incluidas
        if include_target1 and target1_col:
            for _, row in clean_data_for_relations.iterrows():
                source = str(row[source_col])
                # Procesar relación con Destino 1 (Colaboraciones)
                if pd.notna(row[target1_col]):
                    target1 = str(row[target1_col])
                    if source != target1: # Evitar bucles
                        G.add_node(target1) # Asegurar que el destino esté en el grafo
                        target1_nodes.add(target1)
                        # Agregar arista Origen -> Destino 1 (Colaboración)
                        weight = 1.0
                        if weight_col and weight_col in data.columns and pd.notna(row[weight_col]):
                            try:
                                weight = float(row[weight_col])
                            except (ValueError, TypeError):
                                pass
                        if G.has_edge(source, target1):
                            G[source][target1]['weight'] += weight
                        else:
                            G.add_edge(source, target1, weight=weight)

        if include_target2 and target2_col:
            for _, row in clean_data_for_relations.iterrows():
                source = str(row[source_col])
                # Procesar relación con Destino 2 (Conflictos)
                if pd.notna(row[target2_col]):
                    target2 = str(row[target2_col])
                    if source != target2: # Evitar bucles
                        G.add_node(target2) # Asegurar que el destino esté en el grafo
                        target2_nodes.add(target2)
                        # Agregar arista Origen -> Destino 2 (Conflicto)
                        weight2 = 1.0
                        if weight_col and weight_col in data.columns and pd.notna(row[weight_col]):
                            try:
                                weight2 = float(row[weight_col])
                            except (ValueError, TypeError):
                                pass
                        if G.has_edge(source, target2):
                            G[source][target2]['weight'] += weight2
                        else:
                            G.add_edge(source, target2, weight=weight2)

        # --- 4. Diseño (Layout) del grafo ---
        if G.number_of_nodes() == 0:
             return None, "No se pudieron crear nodos a partir de los datos de Origen."

        if layout == 'spring':
            pos = nx.spring_layout(G, k=5/np.sqrt(G.number_of_nodes()), iterations=100, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.circular_layout(G)  # Por defecto

        # --- 5. Definición de colores fijos ---
        COLOR_SOURCE = (255, 215, 0)    # Amarillo (Gold)
        COLOR_COLABORACION = (50, 205, 50)   # Verde (LimeGreen)
        COLOR_CONFLICTO = (220, 20, 60)   # Rojo (Crimson)
        COLOR_EDGE = 'black'            # Color de las aristas
        COLOR_ISOLATED_SOURCE = COLOR_SOURCE # Los orígenes aislados también son amarillos

        # Diccionario para almacenar el color de cada nodo
        node_colors = {}

        # Asignar colores basados en el rol primario del nodo
        # Prioridad: Origen > Colaboración > Conflicto
        # Recalcular sets basados en lo que realmente se procesó
        only_conflicto = set()
        colaboracion_and_or_conflicto = set()
        source_nodes_final = source_nodes_in_graph

        # Si se incluyó target2, calcular solo_conflicto
        if include_target2 and target2_col:
            # Solo conflictos que no son colaboraciones ni orígenes
            only_conflicto = target2_nodes - target1_nodes - source_nodes_in_graph

        # Si se incluyó target1, calcular colaboracion_and_or_conflicto
        if include_target1 and target1_col:
            colaboracion_and_or_conflicto = target1_nodes # Incluye Colaboración solos, Col+Conflicto, Col+Origen

        # Asignar colores
        for node in only_conflicto:
            node_colors[node] = COLOR_CONFLICTO
        for node in colaboracion_and_or_conflicto:
            node_colors[node] = COLOR_COLABORACION # Verde para todos los de Colaboración
        for node in source_nodes_final:
             node_colors[node] = COLOR_SOURCE # Amarillo para todos los de Origen (prioridad más alta)

        # --- 6. Extraer coordenadas y propiedades ---
        node_data_by_type = {
            'Origen': {'x': [], 'y': [], 'text': [], 'sizes': []},
            'Colaboraciones': {'x': [], 'y': [], 'text': [], 'sizes': []},
            'Conflictos': {'x': [], 'y': [], 'text': [], 'sizes': []}
        }

        # Calcular tamaños para todos los nodos que están en el grafo
        node_sizes_dict = {}
        for node in G.nodes():
             # Calcular tamaño del nodo
            if node_size_col and node_size_col in data.columns:
                # Buscar el valor en todas las columnas relevantes (origen, destino1, destino2)
                masks = [data[source_col].astype(str) == node]
                if include_target1 and target1_col:
                    masks.append(data[target1_col].astype(str) == node)
                if include_target2 and target2_col:
                     masks.append(data[target2_col].astype(str) == node)
                combined_mask = pd.concat(masks, axis=1).any(axis=1)
                node_data_for_size = data[combined_mask]
                if not node_data_for_size.empty and pd.api.types.is_numeric_dtype(data[node_size_col]):
                    size_val = node_data_for_size[node_size_col].mean()
                    if pd.notna(size_val):
                        min_size, max_size = 20, 60
                        min_val = data[node_size_col].min()
                        max_val = data[node_size_col].max()
                        if max_val != min_val:
                            normalized_size = min_size + (size_val - min_val) * (max_size - min_size) / (max_val - min_val)
                        else:
                            normalized_size = (min_size + max_size) / 2
                        node_sizes_dict[node] = normalized_size
                    else:
                        node_sizes_dict[node] = 30
                else:
                    node_sizes_dict[node] = 30
            else:
                degree = G.degree(node)
                min_size, max_size = 20, 60
                if G.number_of_nodes() > 1:
                    degrees = [G.degree(n) for n in G.nodes()]
                    min_degree, max_degree = min(degrees), max(degrees)
                    if max_degree != min_degree:
                        normalized_size = min_size + (degree - min_degree) * (max_size - min_size) / (max_degree - min_degree)
                    else:
                        normalized_size = (min_size + max_size) / 2
                else:
                    normalized_size = (min_size + max_size) / 2
                node_sizes_dict[node] = normalized_size

        # Asignar nodos a sus respectivas listas según su tipo/rol para la leyenda
        for node in G.nodes():
            x, y = pos[node]
            size = node_sizes_dict[node]
            # Determinar el tipo para la leyenda basado en el color/rol
            if node in source_nodes_final:
                node_type_key = 'Origen'
            elif node in colaboracion_and_or_conflicto:
                node_type_key = 'Colaboraciones'
            elif node in only_conflicto:
                node_type_key = 'Conflictos'
            else:
                # Nodo sin relación procesada (aislado o no relevante según filtros)
                # Lo asignamos a Origen si es uno, de lo contrario, lo ignoramos o asignamos por defecto.
                # Dado que todos los orígenes se añaden, debería estar en source_nodes_final.
                # Si llega aquí, podría ser un nodo destino que no se procesó.
                # Para simplificar, lo asignamos a Origen.
                node_type_key = 'Origen'

            # Solo añadir a node_data_by_type si el tipo está incluido
            if (node_type_key == 'Origen') or \
               (node_type_key == 'Colaboraciones' and include_target1) or \
               (node_type_key == 'Conflictos' and include_target2):
                node_data_by_type[node_type_key]['x'].append(x)
                node_data_by_type[node_type_key]['y'].append(y)
                node_data_by_type[node_type_key]['text'].append(node)
                node_data_by_type[node_type_key]['sizes'].append(size)

        # --- 7. Crear trazos para las aristas (negras) ---
        # Solo añadir aristas si sus extremos están en el grafo y su tipo está incluido
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=1.5, color=COLOR_EDGE),
            hoverinfo='text',
            text=[],
            mode='lines',
            showlegend=False,
            name='Relaciones'
        )

        # Añadir aristas solo si sus nodos destino están en los sets relevantes
        for source, target, data_edge in G.edges(data=True):
            # Verificar si la arista debe mostrarse según los filtros
            show_edge = False
            # Si la arista va a un nodo de colaboración y se incluyen colaboraciones
            if include_target1 and target in target1_nodes:
                show_edge = True
            # Si la arista va a un nodo de conflicto y se incluyen conflictos
            elif include_target2 and target in target2_nodes:
                show_edge = True

            if show_edge:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
                edge_trace['text'] += (f"{source} -> {target}", "", "")

        # --- 8. Construcción de la figura Plotly ---
        fig = go.Figure()
        # Añadir trazo de aristas
        fig.add_trace(edge_trace)

        # Diccionario de colores para la leyenda
        legend_colors = {
            'Origen': f'rgb{COLOR_SOURCE}',
            'Colaboraciones': f'rgb{COLOR_COLABORACION}',
            'Conflictos': f'rgb{COLOR_CONFLICTO}'
        }

        # Añadir trazos de nodos por tipo para la leyenda
        # Solo si el tipo está incluido y hay nodos de ese tipo
        if include_target1: # Siempre mostrar Origen, pero solo añadir trazo si hay nodos
             if node_data_by_type['Origen']['x']:
                fig.add_trace(go.Scatter(
                    x=node_data_by_type['Origen']['x'], y=node_data_by_type['Origen']['y'],
                    mode='markers',
                    hoverinfo='text',
                    text=node_data_by_type['Origen']['text'],
                    marker=dict(
                        showscale=False,
                        color=legend_colors['Origen'],
                        size=node_data_by_type['Origen']['sizes'],
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    showlegend=True,
                    name='Origen'
                ))
             if node_data_by_type['Colaboraciones']['x']:
                fig.add_trace(go.Scatter(
                    x=node_data_by_type['Colaboraciones']['x'], y=node_data_by_type['Colaboraciones']['y'],
                    mode='markers',
                    hoverinfo='text',
                    text=node_data_by_type['Colaboraciones']['text'],
                    marker=dict(
                        showscale=False,
                        color=legend_colors['Colaboraciones'],
                        size=node_data_by_type['Colaboraciones']['sizes'],
                        line=dict(width=2, color='DarkSlateGrey')
                    ),
                    showlegend=True,
                    name='Colaboraciones'
                ))
        if include_target2 and node_data_by_type['Conflictos']['x']:
             fig.add_trace(go.Scatter(
                x=node_data_by_type['Conflictos']['x'], y=node_data_by_type['Conflictos']['y'],
                mode='markers',
                hoverinfo='text',
                text=node_data_by_type['Conflictos']['text'],
                marker=dict(
                    showscale=False,
                    color=legend_colors['Conflictos'],
                    size=node_data_by_type['Conflictos']['sizes'],
                    line=dict(width=2, color='DarkSlateGrey')
                ),
                showlegend=True,
                name='Conflictos'
            ))


        # Añadir trazo de etiquetas de texto para nodos (opcional, sin leyenda)
        all_node_labels_x = []
        all_node_labels_y = []
        all_node_labels_text = []
        # Recopilar nodos que se van a mostrar (de los trazos añadidos)
        shown_nodes = set()
        for trace in fig.data:
            if trace.mode == 'markers' and hasattr(trace, 'text'):
                 shown_nodes.update(trace.text)

        for node in shown_nodes: # Solo etiquetas para nodos mostrados
             if node in pos: # Asegurar que el nodo tenga posición
                x, y = pos[node]
                all_node_labels_x.append(x)
                all_node_labels_y.append(y - 0.04) # Ajuste de posición
                all_node_labels_text.append(node)

        if all_node_labels_x: # Solo añadir si hay etiquetas
            fig.add_trace(go.Scatter(
                x=all_node_labels_x, y=all_node_labels_y,
                mode='text',
                text=all_node_labels_text,
                textposition="middle center",
                textfont=dict(size=10, color='black'),
                showlegend=False,
                hoverinfo='skip',
                name='Etiquetas'
            ))

        # --- 9. Diseño final del gráfico ---
        title_text = "Sociograma"
        fig.update_layout(
            title=title_text,
            title_x=0.5,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5,
                traceorder='normal',
                font=dict(size=10),
                bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="Gray",
                borderwidth=1
            )
        )
        return fig, None
    except Exception as e:
        import traceback
        # st.error(f"Excepción en create_sociogram: {e}") # Opcional para depuración
        # st.code(traceback.format_exc())
        return None, f"Error al generar el sociograma: {str(e)}"

def show_record_details(record):
    """Muestra los detalles de un registro en una ventana modal"""
    st.markdown("### 📄 Detalles del Registro")
    st.write(record.to_dict())
def generate_pdf_report(filtered_df, plots_data, stats, filename="reporte_analitico.pdf"):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 20, "Reporte Analítico Completo", 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 8, f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.multi_cell(0, 8, f"Total de registros: {len(filtered_df)}")
    pdf.multi_cell(0, 8, f"Total de columnas: {len(filtered_df.columns)}")
    pdf.ln(15)
    pdf.add_section_title("Resumen Ejecutivo")
    pdf.multi_cell(0, 8, "Este reporte contiene un análisis completo de los datos cargados en la aplicación Excel Analytics Pro+, incluyendo visualizaciones, estadísticas descriptivas y los datos filtrados según los criterios seleccionados.")
    pdf.ln(10)
    pdf.add_section_title("Estadísticas Descriptivas")
    pdf.add_table(stats.describe().round(2).reset_index().rename(columns={'index': 'metric'}), "Estadísticas principales")
    pdf.add_section_title("Visualizaciones")
    temp_files = []
    try: # Bloque try para manejar errores de escritura de imagen
        for plot_data in plots_data:
            if plot_data['type'] == 'plot':
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_files.append(temp_file.name)
                    temp_file.close()
                    plot_data['fig'].write_image(
                        temp_file.name,
                        format='png',
                        width=1200, # Ancho en píxeles
                        height=800, # Alto en píxeles
                        scale=3     # Factor de escala (3x = ~300 DPI, 4x = ~400 DPI)
                        )
                    plot_data['fig'].write_image(temp_file.name)
                    pdf.add_plot(temp_file.name, plot_data['caption'])
                except Exception as e:
                    st.error(f"Error al procesar gráfico para PDF: {str(e)}")
            elif plot_data['type'] == 'table':
                pdf.add_table(plot_data['data'], plot_data['caption'])
        pdf.add_page()
        pdf.add_section_title("Datos Filtrados (Muestra)")
        sample_data = filtered_df.head(50).reset_index(drop=True)
        pdf.add_table(sample_data, f"Muestra de {len(sample_data)} registros (de {len(filtered_df)} totales)")
        pdf_bytes = pdf.output(dest='S').encode('latin1')
    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass # Ignorar errores al eliminar
    return pdf_bytes
# --- FUNCIÓN PRINCIPAL ---
def main():
    st.title("🚀 MAPEO DE ACTORES - VALLE CHANCAY - ZAÑA")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Sube tu archivo Excel",
        type=['xlsx', 'xls'],
        key="file_uploader"
    )
    if not uploaded_file:
        st.info("Por favor sube un archivo Excel para comenzar")
        return
    df, load_msg = load_and_process(uploaded_file)
    if not isinstance(df, pd.DataFrame):
        st.error(load_msg)
        return
    st.success(f"✅ Datos cargados: {uploaded_file.name} ({len(df)} registros)")
    filtered_df, filter_error = setup_sidebar_filters(df)
    if not isinstance(filtered_df, pd.DataFrame):
        st.error(filter_error)
        return
    # Inicializar estado del reporte solo si no existe
    if 'report_plots' not in st.session_state:
        st.session_state.report_plots = []
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Datos Filtrados", 
        "📈 Análisis", 
        "📊 Visualizaciones", 
        "蜘️ Radar",
        "🔄 Matriz Poder-Interés",
        "👥 Sociograma",
        "gMaps", # Nueva pestaña para Georreferenciación
        "📑 Generar Reporte"
    ])
    with tab1:
        st.subheader("🔍 Buscador Avanzado")
        search_cols = st.multiselect(
            "Selecciona columnas para buscar",
            filtered_df.columns.tolist(),
            default=filtered_df.columns[:1].tolist()
        )
        search_query = st.text_input("Término de búsqueda", "")
        if search_query and search_cols:
            mask = pd.concat([
                filtered_df[col].astype(str).str.contains(search_query, case=False, na=False) 
                for col in search_cols
            ], axis=1).any(axis=1)
            filtered_df = filtered_df[mask]
            st.success(f"✅ {len(filtered_df)} registros encontrados con '{search_query}'")
        st.subheader("Datos Filtrados")
        st.data_editor(
            filtered_df,
            height=500,
            use_container_width=True,
            key="data_editor",
            num_rows="fixed",
            column_config={
                col: st.column_config.Column(
                    disabled=True
                ) for col in filtered_df.columns
            }
        )
        selected_indices = st.session_state.get("data_editor", {}).get("selected_rows", [])
        if selected_indices:
            st.subheader("📄 Registro Seleccionado")
            selected_record = filtered_df.iloc[selected_indices[0]]
            show_record_details(selected_record)
    with tab2:
        st.subheader("Estadísticas Descriptivas")
        stats = filtered_df.describe()
        st.dataframe(stats, use_container_width=True)
        st.subheader("Resumen de Datos")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Registros", len(filtered_df))
            st.write("Tipos de datos:")
            st.json(filtered_df.dtypes.astype(str).to_dict())
        with col2:
            st.metric("Columnas", len(filtered_df.columns))
            st.write("Valores nulos:")
            st.json(filtered_df.isnull().sum().to_dict())
        # Guardar estadísticas para el reporte
        st.session_state.report_stats = stats
    with tab3:
        plot_type = st.selectbox(
            "Tipo de gráfico",
            ["Histograma", "Dispersión", "Barras", "Cajas"],
            key="plot_selector"
        )
        numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
        cat_cols = filtered_df.select_dtypes(exclude='number').columns.tolist()
        if plot_type == "Histograma" and numeric_cols:
            col = st.selectbox("Columna numérica", numeric_cols)
            fig = px.histogram(filtered_df, x=col, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.report_plots.append({
                'type': 'plot',
                'fig': fig,
                'caption': f"Histograma de {col}"
            })
        elif plot_type == "Dispersión" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Eje X", numeric_cols)
            with col2:
                y_col = st.selectbox("Eje Y", numeric_cols)
            color_col = st.selectbox("Color por", [None] + cat_cols) if cat_cols else None
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.report_plots.append({
                'type': 'plot',
                'fig': fig,
                'caption': f"Dispersión: {x_col} vs {y_col}" + (f" por {color_col}" if color_col else "")
            })
        elif plot_type == "Barras" and cat_cols:
            col = st.selectbox("Categoría", cat_cols)
            if numeric_cols:
                val_col = st.selectbox("Valor", numeric_cols)
                fig = px.bar(filtered_df, x=col, y=val_col)
                caption = f"Barras: {val_col} por {col}"
            else:
                fig = px.bar(filtered_df, x=col)
                caption = f"Conteo por {col}"
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.report_plots.append({
                'type': 'plot',
                'fig': fig,
                'caption': caption
            })
        elif plot_type == "Cajas" and numeric_cols and cat_cols:
            num_col = st.selectbox("Valor numérico", numeric_cols)
            cat_col = st.selectbox("Categoría", cat_cols)
            fig = px.box(filtered_df, x=cat_col, y=num_col)
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.report_plots.append({
                'type': 'plot',
                'fig': fig,
                'caption': f"Diagrama de cajas: {num_col} por {cat_col}"
            })
    with tab4:
        st.subheader("Gráfico de Radar Personalizado")
        numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
        cat_cols = filtered_df.select_dtypes(exclude='number').columns.tolist()
        if len(numeric_cols) >= 1:
            col1, col2 = st.columns(2)
            with col1:
                metrics = st.multiselect(
                    "Selecciona métricas (≥1)",
                    numeric_cols,
                    default=numeric_cols[:1],
                    key="radar_metrics"
                )
            with col2:
                category = st.selectbox(
                    "Agrupar por",
                    [None] + cat_cols,
                    key="radar_category"
                )
            if st.button("Generar Radar", key="generate_radar"):
                if len(metrics) >= 1:
                    with st.spinner("Creando visualización..."):
                        fig, error = create_optimized_radar(filtered_df, metrics, category)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.report_plots.append({
                            'type': 'plot',
                            'fig': fig,
                            'caption': f"Gráfico de Radar: {', '.join(metrics)}" + (f" por {category}" if category else "")
                        })
                        st.subheader("Datos del Radar")
                        if category:
                            radar_data = filtered_df.groupby(category)[metrics].mean()
                            st.dataframe(radar_data)
                            st.session_state.report_plots.append({
                                'type': 'table',
                                'data': radar_data.reset_index(),
                                'caption': f"Promedios por {category}"
                            })
                        else:
                            radar_data = filtered_df[metrics].mean().to_frame("Promedio")
                            st.dataframe(radar_data)
                            st.session_state.report_plots.append({
                                'type': 'table',
                                'data': radar_data.reset_index(),
                                'caption': "Promedios generales"
                            })
                    else:
                        st.error(error)
                else:
                    st.warning("Selecciona al menos 1 métrica")
        else:
            st.warning("Se necesita al menos 1 columna numérica para el radar")
    with tab5:
        st.subheader("Matriz de Poder vs. Interés")
        if isinstance(filtered_df, pd.DataFrame):
            numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
            text_cols = filtered_df.select_dtypes(exclude='number').columns.tolist()
            if len(numeric_cols) >= 2 and len(text_cols) >= 1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    power_col = st.selectbox("Columna de PODER", numeric_cols)
                with col2:
                    interest_col = st.selectbox("Columna de INTERÉS", numeric_cols)
                with col3:
                    stakeholder_col = st.selectbox("Columna de STAKEHOLDERS", text_cols)
                if st.button("Generar Matriz", key="matrix_button"):
                    fig, result = create_power_interest_matrix(
                        filtered_df, power_col, interest_col, stakeholder_col
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.report_plots.append({
                            'type': 'plot',
                            'fig': fig,
                            'caption': f"Matriz Poder-Interés: {power_col} vs {interest_col}"
                        })
                        st.subheader("Resultados por Cuadrante")
                        if isinstance(result, pd.DataFrame):
                            st.write("Distribución de stakeholders por cuadrante:")
                            cuadrante_counts = result['cuadrante'].value_counts().reset_index()
                            cuadrante_counts.columns = ['Cuadrante', 'Cantidad']
                            st.dataframe(cuadrante_counts)
                            st.session_state.report_plots.append({
                                'type': 'table',
                                'data': cuadrante_counts,
                                'caption': "Distribución por cuadrante"
                            })
                            st.write("Detalle completo de clasificación:")
                            st.dataframe(result)
                            st.session_state.report_plots.append({
                                'type': 'table',
                                'data': result,
                                'caption': "Clasificación completa"
                            })
                        else:
                            st.error(result)
                    else:
                        st.error(result)
            else:
                st.warning("Se necesitan al menos 2 columnas numéricas y 1 columna de texto")
        else:
            st.warning("Carga y filtra los datos primero")
    with tab6:
        st.subheader("👥 Sociograma (Diagrama de Red)")
        st.markdown("""
        Un sociograma es una representación visual de las relaciones entre individuos en un grupo.
        Para crearlo, necesitas datos que indiquen quién se relaciona con quién.
        """)
        if isinstance(filtered_df, pd.DataFrame):
            if len(filtered_df.columns) >= 2:
                text_cols = filtered_df.select_dtypes(include=['object', 'string']).columns.tolist()
                numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()

                # Nueva interfaz con dos columnas de destino
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    source_col = st.selectbox("Columna de ORIGEN", text_cols, key="source_col_socio")
                with col2:
                    target1_col = st.selectbox("Columna de DESTINO 1", text_cols, key="target1_col_socio")
                with col3:
                    # target2_col es opcional
                    target2_col = st.selectbox("Columna de DESTINO 2 (opcional)", [None] + text_cols, key="target2_col_socio")
                with col4:
                    layout = st.selectbox("Diseño", ['circular', 'spring', 'random'], key="layout_socio")

                col5, col6 = st.columns(2)
                with col5:
                     weight_col = st.selectbox("Columna de PESO (opcional)", [None] + numeric_cols, key="weight_col_socio")
                with col6:
                    node_size_col = st.selectbox("Tamaño de nodos (opcional)", [None] + numeric_cols + text_cols, key="node_size_col_socio")

                # --- NUEVOS CONTROLES PARA FILTRAR RELACIONES ---
                st.markdown("---")
                st.subheader("Filtrar Relaciones")
                # Usamos st.checkbox para permitir seleccionar qué relaciones incluir
                # Los valores por defecto pueden ser True si la columna existe
                include_target1_default = target1_col is not None
                include_target2_default = target2_col is not None

                col_filter1, col_filter2 = st.columns(2)
                with col_filter1:
                    include_target1 = st.checkbox("Incluir DESTINO 1 (Colaboraciones)", value=include_target1_default, key="include_target1")
                with col_filter2:
                    include_target2 = st.checkbox("Incluir DESTINO 2 (Conflictos)", value=include_target2_default, key="include_target2")
                # --- FIN DE NUEVOS CONTROLES ---

                # Botón para generar el sociograma
                if st.button("Generar Sociograma", key="sociogram_button"):
                    # Verificar que las columnas necesarias estén seleccionadas
                    # Solo se requiere source_col y al menos una de las columnas destino si se va a incluir
                    if not source_col:
                        st.warning("Por favor selecciona la columna de origen.")
                    elif (include_target1 and not target1_col) and (include_target2 and not target2_col):
                         st.warning("Por favor selecciona al menos una columna de destino para incluir.")
                    elif not (include_target1 or include_target2):
                         st.warning("Por favor selecciona al menos un tipo de relación para incluir (DESTINO 1 o DESTINO 2).")
                    else:
                        with st.spinner("Creando sociograma..."):
                            # Llamar a la función actualizada con los nuevos parámetros de filtro
                            fig, error = create_sociogram(
                                filtered_df,
                                source_col,
                                target1_col,
                                target2_col if target2_col != None else None,
                                weight_col if weight_col != None else None,
                                node_size_col if node_size_col != None else None,
                                layout,
                                include_target1, # Pasar el filtro para destino 1
                                include_target2  # Pasar el filtro para destino 2
                            )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            st.session_state.report_plots.append({
                                'type': 'plot',
                                'fig': fig,
                                'caption': f"Sociograma: {source_col}" +
                                           (f" → {target1_col}" if include_target1 and target1_col else "") +
                                           (f" & {target2_col}" if include_target2 and target2_col else "")
                            })
                            # --- Estadísticas de la Red ---
                            # Reutilizamos la lógica de creación del grafo para calcular métricas
                            # Asegurarse de aplicar los mismos filtros aquí
                            temp_data = filtered_df.dropna(subset=[source_col])
                            G_stats = nx.DiGraph()
                            # Añadir todos los orígenes
                            for source_node in set(filtered_df[source_col].dropna().astype(str)):
                                G_stats.add_node(source_node)

                            # Procesar relaciones solo si están incluidas
                            if include_target1 and target1_col:
                                for _, row in temp_data.iterrows():
                                    s = str(row[source_col])
                                    if pd.notna(row[target1_col]):
                                        t1 = str(row[target1_col])
                                        if s != t1:
                                            w = 1.0
                                            if weight_col and weight_col in filtered_df.columns:
                                                try:
                                                    w = float(row[weight_col]) if pd.notna(row[weight_col]) else 1.0
                                                except:
                                                    w = 1.0
                                            if G_stats.has_edge(s, t1):
                                                G_stats[s][t1]['weight'] += w
                                            else:
                                                G_stats.add_edge(s, t1, weight=w)
                            if include_target2 and target2_col:
                                for _, row in temp_data.iterrows():
                                    s = str(row[source_col])
                                    if pd.notna(row[target2_col]):
                                        t2 = str(row[target2_col])
                                        if s != t2: # Evitar bucles
                                            w2 = 1.0
                                            if weight_col and weight_col in filtered_df.columns:
                                                 try:
                                                    w2 = float(row[weight_col]) if pd.notna(row[weight_col]) else 1.0
                                                 except:
                                                    w2 = 1.0
                                            if G_stats.has_edge(s, t2):
                                                G_stats[s][t2]['weight'] += w2
                                            else:
                                                G_stats.add_edge(s, t2, weight=w2)

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Nodos", G_stats.number_of_nodes())
                            col2.metric("Aristas", G_stats.number_of_edges())
                            if G_stats.number_of_nodes() > 1:
                                density = nx.density(G_stats)
                                col3.metric("Densidad", f"{density:.3f}")
                            else:
                                col3.metric("Densidad", "N/A")
                            if G_stats.number_of_nodes() > 0:
                                avg_degree = sum(dict(G_stats.degree()).values()) / G_stats.number_of_nodes()
                                col4.metric("Grado promedio", f"{avg_degree:.2f}")
                        else:
                            st.error(error)
            else:
                st.warning("Se necesitan al menos 2 columnas para crear un sociograma")
        else:
            st.warning("Carga y filtra los datos primero")
    # --- NUEVA PESTAÑA: Georreferenciación ---
    with tab7: #gMaps
        st.subheader("gMaps")
        st.markdown("""
        Visualiza datos geográficos en un mapa interactivo y genera un archivo KML para Google Earth.
        Asegúrate de que tu archivo Excel tenga columnas con **Latitud** y **Longitud** en formato decimal.
        """)
        if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
            potential_lat_cols = [col for col in filtered_df.columns if 'lat' in col.lower() and pd.api.types.is_numeric_dtype(filtered_df[col])]
            potential_lon_cols = [col for col in filtered_df.columns if ('lon' in col.lower() or 'lng' in col.lower()) and pd.api.types.is_numeric_dtype(filtered_df[col])]
            all_numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
            all_cols = filtered_df.columns.tolist()
            col1, col2, col3 = st.columns(3)
            with col1:
                lat_col = st.selectbox(
                    "Columna de LATITUD",
                    all_numeric_cols,
                    index=all_numeric_cols.index(potential_lat_cols[0]) if potential_lat_cols else 0,
                    key="gmaps_lat_col"
                )
            with col2:
                lon_col = st.selectbox(
                    "Columna de LONGITUD",
                    all_numeric_cols,
                    index=all_numeric_cols.index(potential_lon_cols[0]) if potential_lon_cols else (1 if len(all_numeric_cols) > 1 else 0),
                    key="gmaps_lon_col"
                )
            with col3:
                name_col_options = [None] + all_cols
                name_col = st.selectbox(
                    "Columna para NOMBRES (Marcadores)",
                    name_col_options,
                    index=0,
                    key="gmaps_name_col",
                    help="Selecciona una columna para usar como nombre de cada ubicación."
                )
            popup_cols = st.multiselect(
                "Columnas para POPUP (Folium)",
                all_cols,
                default=[],
                key="gmaps_popup_cols",
                help="Selecciona columnas cuyos valores se mostrarán en el popup al hacer clic en un marcador."
            )
            style_col = st.selectbox(
                "Columna para ESTILOS/CAPAS (Folium/KML)",
                [None] + all_cols,
                index=0,
                key="gmaps_style_col",
                help="Selecciona una columna categórica para diferenciar grupos de marcadores por color."
            )
            description_cols = st.multiselect(
                "Columnas para DESCRIPCIÓN (KML)",
                all_cols,
                default=[],
                key="gmaps_desc_cols",
                help="Selecciona columnas cuyos valores se incluirán en la descripción de cada punto en el KML."
            )
            if lat_col and lon_col and lat_col != lon_col:
                 # --- Visualización en Folium Map ---
                st.subheader("📍 Mapa Interactivo (Folium)")
                with st.spinner("Creando mapa interactivo..."):
                    folium_map, map_error = create_folium_map(
                        filtered_df, 
                        lat_col, 
                        lon_col, 
                        name_col if name_col != None else None,
                        popup_cols if popup_cols else None,
                        style_col if style_col != None else None
                    )
                if folium_map:
                    st_folium(folium_map, width=1000, height=600)
                else:
                    st.warning(f"⚠️ {map_error}")
                # --- Generación y Descarga de KML ---
                st.subheader("🌐 Generar Archivo KML")
                kml_filename = st.text_input("Nombre del archivo KML", "Ubicaciones_Mapeadas.kml")
                if not kml_filename.endswith('.kml'):
                     kml_filename += '.kml'
                if st.button("Generar y Descargar KML", key="generate_kml_button"):
                    with st.spinner("Generando archivo KML..."):
                        kml_bytes = generate_kml(
                            filtered_df, 
                            lat_col, 
                            lon_col, 
                            name_col if name_col != None else None,
                            description_cols if description_cols else None,
                            style_col if style_col != None else None,
                            kml_filename
                        )
                        if kml_bytes:
                            st.success("✅ Archivo KML generado!")
                            st.download_button(
                                label="Descargar KML",
                                data=kml_bytes,
                                file_name=kml_filename,
                                mime="application/vnd.google-earth.kml+xml"
                            )
                        else:
                            st.error("❌ Error al generar el archivo KML.")
            else:
                st.warning("⚠️ Por favor selecciona columnas diferentes para Latitud y Longitud.")
        else:
            st.warning("⚠️ Carga y filtra los datos primero. Asegúrate de que el conjunto de datos no esté vacío.")
    with tab8:
        st.subheader("📑 Generar Reporte Completo")
        st.markdown("""
        Genera un reporte PDF con todos los análisis realizados, incluyendo:
        - Gráficos generados
        - Estadísticas descriptivas
        - Datos filtrados (muestra)
        - Resumen ejecutivo
        """)
        report_name = st.text_input("Nombre del reporte", "Reporte_Analitico")
        if st.button("Generar Reporte PDF"):
            if 'report_stats' not in st.session_state:
                st.warning("Realiza al menos un análisis para generar el reporte")
                return
            with st.spinner("Generando reporte PDF..."):
                try:
                    pdf_bytes = generate_pdf_report(
                        filtered_df,
                        st.session_state.report_plots,
                        st.session_state.report_stats,
                        f"{report_name}.pdf"
                    )
                    st.success("✅ Reporte generado con éxito!")
                    st.download_button(
                        label="Descargar Reporte",
                        data=pdf_bytes,
                        file_name=f"{report_name}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Error al generar el reporte: {str(e)}")
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Exportar Datos")
    export_format = st.sidebar.radio(
        "Formato", 
        ["CSV", "Excel", "JSON"],
        horizontal=True
    )
    if st.sidebar.button("Generar Archivo"):
        try:
            if export_format == "CSV":
                csv = filtered_df.to_csv(index=False)
                st.sidebar.download_button(
                    "Descargar CSV",
                    csv,
                    "datos_filtrados.csv",
                    "text/csv"
                )
            elif export_format == "Excel":
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    filtered_df.to_excel(writer, index=False)
                st.sidebar.download_button(
                    "Descargar Excel",
                    output.getvalue(),
                    "datos_filtrados.xlsx",
                    "application/vnd.ms-excel"
                )
            else:
                json_data = filtered_df.to_json(orient="records")
                st.sidebar.download_button(
                    "Descargar JSON",
                    json_data,
                    "datos_filtrados.json",
                    "application/json"
                )
        except Exception as e:
            st.sidebar.error(f"Error al exportar: {str(e)}")
if __name__ == "__main__":
    main()
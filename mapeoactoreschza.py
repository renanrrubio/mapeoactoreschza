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

def create_folium_map(data, lat_col, lon_col, name_col=None, popup_cols=None, style_col=None):
    """
    Crea un mapa interactivo usando Folium.
    """
    try:
        valid_coords_df = data.dropna(subset=[lat_col, lon_col])
        try:
            valid_coords_df = valid_coords_df[pd.to_numeric(valid_coords_df[lat_col], errors='coerce').notnull() &
                                              pd.to_numeric(valid_coords_df[lon_col], errors='coerce').notnull()]
        except Exception:
            pass

        if valid_coords_df.empty:
            return None, "No hay datos válidos para mostrar en el mapa."

        center_lat = valid_coords_df[lat_col].mean()
        center_lon = valid_coords_df[lon_col].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10, control_scale=True)
        
        # --- Corrección: Siempre crear un grupo 'default' ---
        feature_groups = {}
        default_fg = folium.FeatureGroup(name="Todos los puntos")
        default_rgb = (0, 123, 255)
        feature_groups['default'] = {'group': default_fg, 'color': default_rgb}
        default_fg.add_to(m)

        # --- Corrección: Sobrescribir o añadir grupos si hay style_col válida ---
        if style_col and style_col in valid_coords_df.columns:
            unique_styles = valid_coords_df[style_col].dropna().unique()
            if len(unique_styles) > 0: # Solo si hay estilos válidos
                color_palette = px.colors.qualitative.Plotly 

                for i, style_value in enumerate(unique_styles):
                    style_key = str(style_value)
                    hex_color = color_palette[i % len(color_palette)]
                    rgb_color = hex_to_rgb_safe(hex_color) 
                    
                    fg = folium.FeatureGroup(name=style_key)
                    feature_groups[style_key] = {'group': fg, 'color': rgb_color}
                    fg.add_to(m)
                # Si se crearon grupos específicos, el 'default' sigue existiendo pero no se usa necesariamente
            # Si unique_styles está vacío, se queda con el 'default'

        # --- Corrección en el bucle de marcadores ---
        for _, row in valid_coords_df.iterrows():
            try:
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except (ValueError, TypeError):
                continue

            # --- Corrección: Acceder de forma segura ---
            group_key = 'default'
            # Obtener el diccionario del grupo de forma segura
            group_info = feature_groups.get(group_key, feature_groups.get('default', {'color': (0, 123, 255), 'group': list(feature_groups.values())[0]['group'] if feature_groups else default_fg}))
            marker_color = group_info['color']

            if style_col and style_col in valid_coords_df.columns and pd.notna(row[style_col]):
                style_val_key = str(row[style_col])
                if style_val_key in feature_groups:
                    group_key = style_val_key
                    group_info = feature_groups[group_key] # Acceder al grupo correcto
                    marker_color = group_info['color']

            current_fg = group_info['group'] # Usar el grupo del diccionario obtenido

            # ... (resto del código para crear el marcador) ...
            popup_content = ""
            if popup_cols:
                 popup_lines = [f"<b>{col}:</b> {row[col]}" for col in popup_cols if col in valid_coords_df.columns and pd.notna(row[col])]
                 popup_content = "<br>".join(popup_lines)
            else:
                popup_name = str(row[name_col]) if name_col and name_col in valid_coords_df.columns and pd.notna(row[name_col]) else f"Punto ({lat:.4f}, {lon:.4f})"
                popup_content = f"<b>{popup_name}</b>"

            tooltip_text = str(row[name_col]) if name_col and name_col in valid_coords_df.columns and pd.notna(row[name_col]) else f"ID: {_}"

            r, g, b = marker_color
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            icon_color_str = 'white' if brightness < 128 else 'black'

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=tooltip_text,
                icon=folium.Icon(color='lightgray', icon_color=icon_color_str, icon='map-marker', prefix='fa')
            ).add_to(current_fg) # Añadir al grupo correcto

        if len(feature_groups) > 1: # Solo mostrar control si hay más de un grupo
            folium.LayerControl().add_to(m)

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
            title = f"Radar por {category}"
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
                orientation="h",
                yanchor="bottom",
                y=1.15,
                xanchor="center",
                x=0.5
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

def create_power_interest_matrix(data, power_col, interest_col, stakeholder_col):
    """Genera matriz de Poder vs. Interés con cuadrantes mejorados y resultados"""
    try:
        required_cols = [power_col, interest_col, stakeholder_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return None, f"Columnas faltantes: {', '.join(missing_cols)}"
        fig = go.Figure()
        max_power = data[power_col].max() * 1.1
        max_interest = data[interest_col].max() * 1.1
        power_mid = max_power / 2
        interest_mid = max_interest / 2
        data['cuadrante'] = data.apply(
            lambda row: (
                "Gestionar Activamente" if (row[power_col] >= power_mid and row[interest_col] >= interest_mid) else
                "Monitorear" if (row[power_col] < power_mid and row[interest_col] >= interest_mid) else
                "Mantener Satisfechos" if (row[power_col] >= power_mid and row[interest_col] < interest_mid) else
                "Mantener Informados"
            ), axis=1
        )
        for cuadrante, color in zip(
            ["Gestionar Activamente", "Monitorear", "Mantener Satisfechos", "Mantener Informados"],
            ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
        ):
            df_cuadrante = data[data['cuadrante'] == cuadrante]
            if not df_cuadrante.empty:
                fig.add_trace(go.Scatter(
                    x=df_cuadrante[power_col],
                    y=df_cuadrante[interest_col],
                    mode='markers',
                    name=cuadrante,
                    marker=dict(
                        size=12,
                        color=color,
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    hoverinfo='text',
                    hovertext=df_cuadrante.apply(
                        lambda row: f"<b>{row[stakeholder_col]}</b><br>Poder: {row[power_col]}<br>Interés: {row[interest_col]}<br>Cuadrante: {row['cuadrante']}",
                        axis=1
                    )
                ))
        fig.add_shape(type="line", 
                     x0=power_mid, y0=0, x1=power_mid, y1=max_interest, 
                     line=dict(color="gray", width=2, dash="dot"))
        fig.add_shape(type="line", 
                     x0=0, y0=interest_mid, x1=max_power, y1=interest_mid,
                     line=dict(color="gray", width=2, dash="dot"))
        fig.add_shape(type="rect",
            x0=0, y0=interest_mid, x1=power_mid, y1=max_interest,
            fillcolor="rgba(173, 216, 230, 0.1)",
            line=dict(width=0)
        )
        fig.add_shape(type="rect",
            x0=power_mid, y0=interest_mid, x1=max_power, y1=max_interest,
            fillcolor="rgba(144, 238, 144, 0.1)",
            line=dict(width=0)
        )
        fig.add_shape(type="rect",
            x0=0, y0=0, x1=power_mid, y1=interest_mid,
            fillcolor="rgba(255, 228, 181, 0.1)",
            line=dict(width=0)
        )
        fig.add_shape(type="rect",
            x0=power_mid, y0=0, x1=max_power, y1=interest_mid,
            fillcolor="rgba(255, 182, 193, 0.1)",
            line=dict(width=0)
        )
        counts = data['cuadrante'].value_counts().to_dict()
        fig.update_layout(
            title="Matriz de Poder vs. Interés",
            xaxis_title="Poder",
            yaxis_title="Interés",
            xaxis=dict(range=[0, max_power]),
            yaxis=dict(range=[0, max_interest]),
            height=600,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,240,240,0.8)',
            annotations=[
                dict(
                    x=0.25, y=0.95, xref="paper", yref="paper",
                    text=f"<b>MONITOREAR</b><br>({counts.get('Monitorear', 0)} elementos)",
                    showarrow=False,
                    font=dict(size=12, color="darkblue"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="darkblue",
                    borderwidth=1,
                    borderpad=4
                ),
                dict(
                    x=0.75, y=0.95, xref="paper", yref="paper",
                    text=f"<b>GESTIONAR ACTIVAMENTE</b><br>({counts.get('Gestionar Activamente', 0)} elementos)",
                    showarrow=False,
                    font=dict(size=12, color="darkgreen"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="darkgreen",
                    borderwidth=1,
                    borderpad=4
                ),
                dict(
                    x=0.25, y=0.05, xref="paper", yref="paper",
                    text=f"<b>MANTENER INFORMADOS</b><br>({counts.get('Mantener Informados', 0)} elementos)",
                    showarrow=False,
                    font=dict(size=12, color="darkorange"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="darkorange",
                    borderwidth=1,
                    borderpad=4
                ),
                dict(
                    x=0.75, y=0.05, xref="paper", yref="paper",
                    text=f"<b>MANTENER SATISFECHOS</b><br>({counts.get('Mantener Satisfechos', 0)} elementos)",
                    showarrow=False,
                    font=dict(size=12, color="firebrick"),
                    bgcolor="rgba(255,255,255,0.7)",
                    bordercolor="firebrick",
                    borderwidth=1,
                    borderpad=4
                )
            ],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        return fig, data[['cuadrante', stakeholder_col, power_col, interest_col]].sort_values(
            by=['cuadrante', power_col, interest_col], ascending=[True, False, False]
        )
    except Exception as e:
        return None, f"Error al generar matriz: {str(e)}"

def create_sociogram(data, source_col, target_col, weight_col=None, node_size_col=None, layout='circular'):
    """
    Crea un sociograma (diagrama de red) a partir de datos de relaciones.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            return None, "Los datos deben ser un DataFrame de pandas"
        if data.empty:
            return None, "El conjunto de datos está vacío"
        required_cols = [source_col, target_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return None, f"Faltan columnas requeridas: {', '.join(missing_cols)}"
        clean_data = data.dropna(subset=[source_col, target_col])
        if clean_data.empty:
            return None, "No hay datos válidos después de eliminar filas con valores nulos"
        G = nx.DiGraph()
        for _, row in clean_data.iterrows():
            source = str(row[source_col])
            target = str(row[target_col])
            if source == target:
                continue
            G.add_node(source)
            G.add_node(target)
            if weight_col and weight_col in data.columns:
                try:
                    weight = float(row[weight_col]) if pd.notna(row[weight_col]) else 1.0
                except (ValueError, TypeError):
                    weight = 1.0
            else:
                weight = 1.0
            if G.has_edge(source, target):
                G[source][target]['weight'] += weight
            else:
                G.add_edge(source, target, weight=weight)
        if G.number_of_nodes() == 0:
            return None, "No se pudieron crear nodos a partir de los datos"
        
        # --- Mejora en el layout para reducir cruces ---
        if layout == 'spring':
            # Aumentar iteraciones y ajustar k para separar más los nodos
            pos = nx.spring_layout(G, k=5/np.sqrt(G.number_of_nodes()), iterations=100, seed=42) 
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.circular_layout(G)  # Por defecto
        
        # --- Generar colores únicos para nodos ---
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        hues = np.linspace(0, 1, num_nodes, endpoint=False)
        node_colors = {}
        for i, node in enumerate(nodes):
            rgb = colorsys.hsv_to_rgb(hues[i], 0.8, 0.8)
            node_colors[node] = tuple(int(c * 255) for c in rgb)

        # Extraer coordenadas y propiedades
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        node_labels_x = []
        node_labels_y = []
        node_colors_plotly = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_colors_plotly.append(f'rgb{node_colors[node]}')
            node_labels_x.append(x)
            node_labels_y.append(y - 0.04)
            if node_size_col and node_size_col in data.columns:
                node_data = data[(data[source_col].astype(str) == node) | (data[target_col].astype(str) == node)]
                if not node_data.empty and pd.api.types.is_numeric_dtype(data[node_size_col]):
                    size_val = node_data[node_size_col].mean()
                    if pd.notna(size_val):
                        min_size, max_size = 20, 60
                        min_val = data[node_size_col].min()
                        max_val = data[node_size_col].max()
                        if max_val != min_val:
                            normalized_size = min_size + (size_val - min_val) * (max_size - min_size) / (max_val - min_val)
                        else:
                            normalized_size = (min_size + max_size) / 2
                        node_sizes.append(normalized_size)
                    else:
                        node_sizes.append(30)
                else:
                    node_sizes.append(30)
            else:
                degree = G.degree(node)
                min_size, max_size = 20, 60
                if G.number_of_nodes() > 1:
                    min_degree = min([G.degree(n) for n in G.nodes()])
                    max_degree = max([G.degree(n) for n in G.nodes()])
                    if max_degree != min_degree:
                        normalized_size = min_size + (degree - min_degree) * (max_size - min_size) / (max_degree - min_degree)
                    else:
                        normalized_size = (min_size + max_size) / 2
                else:
                    normalized_size = (min_size + max_size) / 2
                node_sizes.append(normalized_size)

        # Crear trazos para las aristas con colores
        edge_traces = []
        edge_colors = px.colors.qualitative.Plotly
        
        for i, (source, target, data_edge) in enumerate(G.edges(data=True)):
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            source_index = nodes.index(source) if source in nodes else 0
            edge_color = edge_colors[source_index % len(edge_colors)]

            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1.5, color=edge_color),
                hoverinfo='text',
                text=f"{source} -> {target}",
                mode='lines',
                showlegend=False,
                name=f"Edge {i}"
            )
            edge_traces.append(edge_trace)

        fig = go.Figure()
        for trace in edge_traces:
             fig.add_trace(trace)
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=False,
                color=node_colors_plotly,
                size=node_sizes,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            showlegend=False,
            name='Nodos'
        ))
        fig.add_trace(go.Scatter(
            x=node_labels_x, y=node_labels_y,
            mode='text',
            text=node_text,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.update_layout(
            title="Sociograma",
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        return fig, None
    except Exception as e:
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
        "🕷️ Radar",
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
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    source_col = st.selectbox("Columna de ORIGEN", text_cols, key="source_col")
                with col2:
                    target_col = st.selectbox("Columna de DESTINO", text_cols, key="target_col")
                with col3:
                    weight_col = st.selectbox("Columna de PESO (opcional)", [None] + numeric_cols, key="weight_col")
                with col4:
                    layout = st.selectbox("Diseño", ['circular', 'spring', 'random'], key="layout")
                node_size_col = st.selectbox("Tamaño de nodos (opcional)", [None] + numeric_cols + text_cols, key="node_size_col")
                if st.button("Generar Sociograma", key="sociogram_button"):
                    if source_col and target_col:
                        with st.spinner("Creando sociograma..."):
                            fig, error = create_sociogram(
                                filtered_df, 
                                source_col, 
                                target_col, 
                                weight_col if weight_col != None else None,
                                node_size_col if node_size_col != None else None,
                                layout
                            )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            st.session_state.report_plots.append({
                                'type': 'plot',
                                'fig': fig,
                                'caption': f"Sociograma: {source_col} → {target_col}"
                            })
                            st.subheader("Estadísticas de la Red")
                            temp_data = filtered_df.dropna(subset=[source_col, target_col])
                            G = nx.DiGraph()
                            for _, row in temp_data.iterrows():
                                s, t = str(row[source_col]), str(row[target_col])
                                if s != t:
                                    if weight_col and weight_col in filtered_df.columns:
                                        try:
                                            w = float(row[weight_col]) if pd.notna(row[weight_col]) else 1.0
                                        except:
                                            w = 1.0
                                    else:
                                        w = 1.0
                                    if G.has_edge(s, t):
                                        G[s][t]['weight'] += w
                                    else:
                                        G.add_edge(s, t, weight=w)
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Nodos", G.number_of_nodes())
                            col2.metric("Aristas", G.number_of_edges())
                            if G.number_of_nodes() > 1:
                                density = nx.density(G)
                                col3.metric("Densidad", f"{density:.3f}")
                            else:
                                col3.metric("Densidad", "N/A")
                            if G.number_of_nodes() > 0:
                                avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                                col4.metric("Grado promedio", f"{avg_degree:.2f}")
                        else:
                            st.error(error)
                    else:
                        st.warning("Por favor selecciona las columnas de origen y destino")
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
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Excel Analytics Pro+",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache de datos
@st.cache_data(show_spinner="Cargando y procesando datos...")
def load_and_process(file, sheet_name=None):
    """Carga y valida el archivo Excel"""
    try:
        if not file:
            return None, "No se cargó ningún archivo"
        
        if file.size > 200 * 1024 * 1024:  # 200MB límite
            return None, "Archivo demasiado grande (límite: 200MB)"
        
        engines = ['openpyxl', 'xlrd']
        for engine in engines:
            try:
                excel_data = pd.ExcelFile(BytesIO(file.read()), engine=engine)
                df = pd.read_excel(excel_data, sheet_name=sheet_name) if sheet_name else pd.read_excel(excel_data)
                return df, "Datos cargados exitosamente"
            except Exception as e:
                continue
        
        return None, "No se pudo leer el archivo con ningún motor disponible"
    except Exception as e:
        return None, f"Error crítico: {str(e)}"

# Función de radar modificada (≥1 métrica, solo líneas, rango mínimo=1)
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
                    range=[1, data[valid_metrics].max().max() * 1.1],  # Rango mínimo=1
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

# Función para matriz Poder vs. Interés (ACTUALIZADA)
def create_power_interest_matrix(data, power_col, interest_col, stakeholder_col):
    """Genera matriz de Poder vs. Interés con cuadrantes mejorados y resultados"""
    try:
        required_cols = [power_col, interest_col, stakeholder_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return None, f"Columnas faltantes: {', '.join(missing_cols)}"

        fig = go.Figure()

        # Calcular máximos para los ejes
        max_power = data[power_col].max() * 1.1
        max_interest = data[interest_col].max() * 1.1
        power_mid = max_power / 2
        interest_mid = max_interest / 2

        # Clasificar stakeholders por cuadrante
        data['cuadrante'] = data.apply(
            lambda row: (
                "Gestionar Activamente" if (row[power_col] >= power_mid and row[interest_col] >= interest_mid) else
                "Monitorear" if (row[power_col] < power_mid and row[interest_col] >= interest_mid) else
                "Mantener Satisfechos" if (row[power_col] >= power_mid and row[interest_col] < interest_mid) else
                "Mantener Informados"
            ), axis=1
        )

        # Scatter plot con colores por cuadrante
        for cuadrante, color in zip(
            ["Gestionar Activamente", "Monitorear", "Mantener Satisfechos", "Mantener Informados"],
            ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']  # Verde, Azul, Naranja, Rojo
        ):
            df_cuadrante = data[data['cuadrante'] == cuadrante]
            if not df_cuadrante.empty:
                fig.add_trace(go.Scatter(
                    x=df_cuadrante[power_col],
                    y=df_cuadrante[interest_col],
                    mode='markers',  # Solo marcadores, sin texto
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

        # Líneas divisorias mejoradas
        fig.add_shape(type="line", 
                     x0=power_mid, y0=0, x1=power_mid, y1=max_interest, 
                     line=dict(color="gray", width=2, dash="dot"))
        fig.add_shape(type="line", 
                     x0=0, y0=interest_mid, x1=max_power, y1=interest_mid,
                     line=dict(color="gray", width=2, dash="dot"))

        # Fondos semitransparentes para cada cuadrante
        fig.add_shape(type="rect",
            x0=0, y0=interest_mid, x1=power_mid, y1=max_interest,
            fillcolor="rgba(173, 216, 230, 0.1)",  # Azul claro
            line=dict(width=0)
        )
        fig.add_shape(type="rect",
            x0=power_mid, y0=interest_mid, x1=max_power, y1=max_interest,
            fillcolor="rgba(144, 238, 144, 0.1)",  # Verde claro
            line=dict(width=0)
        )
        fig.add_shape(type="rect",
            x0=0, y0=0, x1=power_mid, y1=interest_mid,
            fillcolor="rgba(255, 228, 181, 0.1)",  # Naranja claro
            line=dict(width=0)
        )
        fig.add_shape(type="rect",
            x0=power_mid, y0=0, x1=max_power, y1=interest_mid,
            fillcolor="rgba(255, 182, 193, 0.1)",  # Rosa claro
            line=dict(width=0)
        )

        # Contar elementos por cuadrante
        counts = data['cuadrante'].value_counts().to_dict()
        
        # Configuración del layout con anotaciones flotantes
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
                # Cuadrante Monitorear (arriba izquierda)
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
                # Cuadrante Gestionar Activamente (arriba derecha)
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
                # Cuadrante Mantener Informados (abajo izquierda)
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
                # Cuadrante Mantener Satisfechos (abajo derecha)
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

def main():
    st.title("🚀 Excel Analytics Pro+")
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Datos Filtrados", 
        "📈 Análisis", 
        "📊 Visualizaciones", 
        "🕷️ Radar",
        "🔄 Matriz Poder-Interés"
    ])
    
    with tab1:
        st.dataframe(filtered_df, height=500, use_container_width=True)
    
    with tab2:
        st.subheader("Estadísticas Descriptivas")
        st.dataframe(filtered_df.describe(), use_container_width=True)
        
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
        
        elif plot_type == "Dispersión" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Eje X", numeric_cols)
            with col2:
                y_col = st.selectbox("Eje Y", numeric_cols)
            color_col = st.selectbox("Color por", [None] + cat_cols) if cat_cols else None
            fig = px.scatter(filtered_df, x=x_col, y=y_col, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Barras" and cat_cols:
            col = st.selectbox("Categoría", cat_cols)
            if numeric_cols:
                val_col = st.selectbox("Valor", numeric_cols)
                fig = px.bar(filtered_df, x=col, y=val_col)
            else:
                fig = px.bar(filtered_df, x=col)
            st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Cajas" and numeric_cols and cat_cols:
            num_col = st.selectbox("Valor numérico", numeric_cols)
            cat_col = st.selectbox("Categoría", cat_cols)
            fig = px.box(filtered_df, x=cat_col, y=num_col)
            st.plotly_chart(fig, use_container_width=True)
    
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
                        
                        st.subheader("Datos del Radar")
                        if category:
                            st.dataframe(filtered_df.groupby(category)[metrics].mean())
                        else:
                            st.dataframe(filtered_df[metrics].mean().to_frame("Promedio"))
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
                        
                        st.subheader("Resultados por Cuadrante")
                        if isinstance(result, pd.DataFrame):
                            # Mostrar resumen por cuadrante
                            st.write("Distribución de stakeholders por cuadrante:")
                            cuadrante_counts = result['cuadrante'].value_counts().reset_index()
                            cuadrante_counts.columns = ['Cuadrante', 'Cantidad']
                            st.dataframe(cuadrante_counts)
                            
                            # Mostrar detalles por cuadrante
                            st.write("Detalle completo de clasificación:")
                            st.dataframe(result)
                        else:
                            st.error(result)
                    else:
                        st.error(result)
            else:
                st.warning("Se necesitan al menos 2 columnas numéricas y 1 columna de texto")
        else:
            st.warning("Carga y filtra los datos primero")
    
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
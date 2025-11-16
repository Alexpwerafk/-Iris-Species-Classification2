#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proyecto de Clasificación de Especies de Iris
Universidad de la Costa - Data Mining 2024
Desarrollado por: Alexander Gutierrez

Este proyecto implementa un clasificador de especies de Iris utilizando Random Forest
con optimización mediante GridSearchCV y visualizaciones interactivas en Streamlit.
"""

# ===== IMPORTS ORDENADOS Y COMENTADOS =====
import streamlit as st  # Framework web para dashboard interactivo
import pandas as pd  # Manipulación de datos
import numpy as np  # Operaciones numéricas
import matplotlib.pyplot as plt  # Visualizaciones básicas
import seaborn as sns  # Visualizaciones estadísticas avanzadas
import plotly.express as px  # Visualizaciones interactivas
import plotly.graph_objects as go  # Gráficos personalizados con Plotly
from sklearn.datasets import load_iris  # Carga del dataset Iris
from sklearn.model_selection import train_test_split, GridSearchCV  # División y optimización
from sklearn.preprocessing import StandardScaler  # Normalización de características
from sklearn.ensemble import RandomForestClassifier  # Modelo de clasificación
from sklearn.pipeline import Pipeline  # Pipeline para flujo de trabajo ML
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report)  # Métricas de evaluación
from sklearn.decomposition import PCA  # Análisis de componentes principales
import warnings  # Manejo de advertencias
warnings.filterwarnings('ignore')  # Ignorar advertencias para mejor UX

# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(
    page_title="🌸 Iris Classifier Pro",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Proyecto de Clasificación de Especies de Iris - Universidad de la Costa 2024"
    }
)

# ===== SISTEMA DE CACHÉ AVANZADO =====
@st.cache_data(ttl=3600)
def load_and_explore_data():
    """
    Carga y explora el dataset Iris con estadísticas descriptivas.
    Utiliza cache para optimizar performance (ttl=3600 segundos).
    
    Returns:
        tuple: (DataFrame de características, Series de etiquetas)
    """
    # Cargar dataset de sklearn
    iris = load_iris()
    
    # Crear DataFrame con nombres de características
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    df = pd.DataFrame(iris.data, columns=feature_names)
    df['Species'] = iris.target_names[iris.target]
    
    # Separar características y etiquetas
    X = df[feature_names]
    y = df['Species']
    
    return X, y, iris

@st.cache_resource
def create_ml_pipeline():
    """
    Crea pipeline ML completo con preprocesamiento y modelo Random Forest.
    
    Returns:
        Pipeline: Pipeline configurado con StandardScaler y RandomForestClassifier
    """
    # Definir hiperparámetros para GridSearchCV
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Crear pipeline con preprocesamiento y clasificador
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline, param_grid

def train_and_evaluate_model(pipeline, param_grid, X, y):
    """
    Entrena y evalúa el modelo con GridSearchCV.
    
    Args:
        pipeline: Pipeline de ML
        param_grid: Grid de hiperparámetros
        X: Características
        y: Etiquetas
        
    Returns:
        dict: Resultados de evaluación y modelo entrenado
    """
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # GridSearchCV para optimización
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Entrenar modelo
    with st.spinner('🤖 Entrenando modelo con GridSearchCV...'):
        grid_search.fit(X_train, y_train)
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    
    # Predicciones
    y_pred = best_model.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'model': best_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return metrics

def get_feature_importance(pipeline):
    """
    Obtiene la importancia de las características del modelo entrenado.
    
    Args:
        pipeline: Pipeline con modelo entrenado
        
    Returns:
        pd.DataFrame: Importancia de características ordenadas
    """
    # Obtener importancias del clasificador
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    # Crear DataFrame ordenado
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    return importance_df

def create_3d_visualization(data, target, new_sample=None):
    """
    Crea visualización 3D interactiva con PCA.
    
    Args:
        data: Datos de características
        target: Etiquetas de especies
        new_sample: Muestra nueva para visualizar (opcional)
        
    Returns:
        go.Figure: Figura de Plotly 3D
    """
    # Aplicar PCA para reducción dimensional
    pca = PCA(n_components=3)
    data_3d = pca.fit_transform(data)
    
    # Crear DataFrame para Plotly
    df_3d = pd.DataFrame(data_3d, columns=['PC1', 'PC2', 'PC3'])
    df_3d['Species'] = target.values
    
    # Colores para cada especie
    colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
    
    # Crear figura 3D
    fig = go.Figure()
    
    # Agregar puntos para cada especie
    for species in df_3d['Species'].unique():
        species_data = df_3d[df_3d['Species'] == species]
        fig.add_trace(go.Scatter3d(
            x=species_data['PC1'],
            y=species_data['PC2'],
            z=species_data['PC3'],
            mode='markers',
            name=species,
            marker=dict(
                size=8,
                color=colors[species],
                opacity=0.8
            ),
            hovertemplate=f'<b>{species}</b><br>' +
                         'PC1: %{x:.2f}<br>' +
                         'PC2: %{y:.2f}<br>' +
                         'PC3: %{z:.2f}<extra></extra>'
        ))
    
    # Si hay nueva muestra, agregarla
    if new_sample is not None:
        new_sample_3d = pca.transform(new_sample.reshape(1, -1))
        fig.add_trace(go.Scatter3d(
            x=[new_sample_3d[0, 0]],
            y=[new_sample_3d[0, 1]],
            z=[new_sample_3d[0, 2]],
            mode='markers',
            name='Nueva Muestra',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='<b>Nueva Muestra</b><br>' +
                         'PC1: %{x:.2f}<br>' +
                         'PC2: %{y:.2f}<br>' +
                         'PC3: %{z:.2f}<extra></extra>'
        ))
    
    # Configurar layout
    fig.update_layout(
        title='Visualización 3D con PCA (95% de varianza explicada)',
        scene=dict(
            xaxis_title='PC1 (%.1f%%)' % (pca.explained_variance_ratio_[0] * 100),
            yaxis_title='PC2 (%.1f%%)' % (pca.explained_variance_ratio_[1] * 100),
            zaxis_title='PC3 (%.1f%%)' % (pca.explained_variance_ratio_[2] * 100),
            bgcolor='rgba(0,0,0,0)'
        ),
        width=800,
        height=600,
        legend=dict(x=0.7, y=0.9)
    )
    
    return fig

def predict_species(pipeline, sepal_length, sepal_width, petal_length, petal_width):
    """
    Predice la especie de iris basada en las características ingresadas.
    
    Args:
        pipeline: Modelo entrenado
        sepal_length: Longitud del sépalo
        sepal_width: Ancho del sépalo
        petal_length: Longitud del pétalo
        petal_width: Ancho del pétalo
        
    Returns:
        tuple: (especie_predicha, probabilidades)
    """
    # Crear array de características
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predicción
    prediction = pipeline.predict(features)[0]
    
    # Probabilidades (solo si el modelo las soporta)
    try:
        probabilities = pipeline.predict_proba(features)[0]
        prob_dict = dict(zip(pipeline.classes_, probabilities))
    except:
        prob_dict = {prediction: 1.0}
    
    return prediction, prob_dict

# ===== LÓGICA PRINCIPAL =====
def main():
    # CSS personalizado para header profesional
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .prediction-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header profesional
    st.markdown("""
    <div class="main-header">
        <h1>🌸 Iris Classifier Pro</h1>
        <h3>Machine Learning Avanzado con Random Forest</h3>
        <p>Universidad de la Costa - Data Mining 2024</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar datos
    X, y, iris = load_and_explore_data()
    
    # Sidebar con controles
    st.sidebar.header("🎛️ Controles de Predicción")
    
    # Sliders para características
    with st.sidebar.form("prediction_form"):
        st.write("### 🔬 Ingresa las características:")
        
        sepal_length = st.slider(
            "Sepal Length (cm)", 
            min_value=float(X['Sepal Length'].min()),
            max_value=float(X['Sepal Length'].max()),
            value=float(X['Sepal Length'].mean()),
            step=0.1
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)", 
            min_value=float(X['Sepal Width'].min()),
            max_value=float(X['Sepal Width'].max()),
            value=float(X['Sepal Width'].mean()),
            step=0.1
        )
        
        petal_length = st.slider(
            "Petal Length (cm)", 
            min_value=float(X['Petal Length'].min()),
            max_value=float(X['Petal Length'].max()),
            value=float(X['Petal Length'].mean()),
            step=0.1
        )
        
        petal_width = st.slider(
            "Petal Width (cm)", 
            min_value=float(X['Petal Width'].min()),
            max_value=float(X['Petal Width'].max()),
            value=float(X['Petal Width'].mean()),
            step=0.1
        )
        
        predict_button = st.form_submit_button("🚀 Predecir Especie", use_container_width=True)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔬 Análisis Exploratorio", "🌍 Visualización 3D", "🔮 Predicción"])
    
    # Crear pipeline y entrenar modelo
    pipeline, param_grid = create_ml_pipeline()
    metrics = train_and_evaluate_model(pipeline, param_grid, X, y)
    
    # TAB 1: Dashboard
    with tab1:
        st.header("📊 Dashboard de Métricas")
        
        # Métricas principales con st.metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("🎯 Precision", f"{metrics['precision']:.3f}")
        with col3:
            st.metric("🎯 Recall", f"{metrics['recall']:.3f}")
        with col4:
            st.metric("🎯 F1-Score", f"{metrics['f1']:.3f}")
        
        # Barras de progreso coloridas para métricas
        st.write("### 📈 Barras de Progreso de Métricas")
        
        metrics_cols = st.columns(4)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (name, value, color) in enumerate(zip(metric_names, metric_values, colors)):
            with metrics_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{name}</h4>
                    <div style="background-color: {color}; width: {value*100}%; height: 20px; 
                         border-radius: 10px; margin: 10px 0;"></div>
                    <p>{value:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Importancia de características
        st.write("### 🎯 Importancia de Características")
        importance_df = get_feature_importance(metrics['model'])
        
        fig_importance = px.bar(
            importance_df, x='Importance', y='Feature',
            orientation='h', title='Importancia de Características en Random Forest',
            color='Importance', color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Matriz de confusión interactiva
        st.write("### 🔥 Matriz de Confusión")
        cm = metrics['confusion_matrix']
        
        fig_cm = px.imshow(
            cm, text_auto=True, aspect='auto',
            x=['Setosa', 'Versicolor', 'Virginica'],
            y=['Setosa', 'Versicolor', 'Virginica'],
            color_continuous_scale='Blues',
            title='Matriz de Confusión - Modelo Random Forest'
        )
        fig_cm.update_xaxes(title="Predicción")
        fig_cm.update_yaxes(title="Real")
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Mejores parámetros
        st.write("### ⚙️ Mejores Hiperparámetros")
        st.json(metrics['best_params'])
    
    # TAB 2: Análisis Exploratorio
    with tab2:
        st.header("🔬 Análisis Exploratorio de Datos")
        
        # Estadísticas descriptivas
        st.write("### 📊 Estadísticas Descriptivas")
        st.dataframe(X.describe())
        
        # Crear DataFrame completo para visualizaciones
        df_viz = X.copy()
        df_viz['Species'] = y
        
        # Histogramas por clase
        st.write("### 📊 Distribuciones por Clase")
        fig_hist = px.histogram(
            df_viz.melt(id_vars=['Species'], var_name='Feature', value_name='Value'),
            x='Value', color='Species', facet_col='Feature',
            title='Histogramas de Características por Especie',
            color_discrete_map={'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Scatter matrix (pairplot)
        st.write("### 🔍 Scatter Matrix (Pairplot)")
        fig_pairplot = sns.pairplot(
            df_viz, hue='Species', 
            palette={'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
        )
        st.pyplot(fig_pairplot)
        
        # Violin plots
        st.write("### 🎻 Violin Plots - Distribuciones de Características")
        for feature in X.columns:
            fig_violin = px.violin(
                df_viz, y=feature, color='Species',
                title=f'Distribución de {feature} por Especie',
                color_discrete_map={'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    
    # TAB 3: Visualización 3D
    with tab3:
        st.header("🌍 Visualización 3D con PCA")
        
        st.write("""
        ### 🔬 Análisis de Componentes Principales (PCA)
        
        La visualización 3D utiliza PCA para reducir la dimensionalidad de las 4 características 
        a 3 componentes principales, manteniendo el 95% de la varianza explicada del dataset original.
        """
        )
        
        # Crear visualización 3D
        fig_3d = create_3d_visualization(X, y)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Explicación de PCA
        pca = PCA(n_components=3)
        pca.fit(X)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### 📊 Varianza Explicada por Componente")
            variance_df = pd.DataFrame({
                'Componente': ['PC1', 'PC2', 'PC3'],
                'Varianza (%)': pca.explained_variance_ratio_ * 100
            })
            st.dataframe(variance_df)
        
        with col2:
            st.write("### 🎯 Varianza Total Explicada")
            total_variance = sum(pca.explained_variance_ratio_) * 100
            st.metric("Varianza Total", f"{total_variance:.1f}%")
    
    # TAB 4: Predicción
    if predict_button:
        with tab4:
            st.header("🔮 Resultado de Predicción")
            
            # Realizar predicción
            prediction, probabilities = predict_species(
                metrics['model'], sepal_length, sepal_width, petal_length, petal_width
            )
            
            # Mostrar resultado con emoji
            species_emojis = {
                'setosa': '🌺',
                'versicolor': '🌸',
                'virginica': '🌼'
            }
            
            st.markdown(f"""
            <div class="prediction-result">
                <h2>{species_emojis.get(prediction, '🌸')} Especie Predicha: {prediction.title()}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Probabilidades
            st.write("### 📊 Probabilidades por Especie")
            prob_df = pd.DataFrame(list(probabilities.items()), columns=['Especie', 'Probabilidad'])
            prob_df['Probabilidad (%)'] = prob_df['Probabilidad'] * 100
            
            fig_prob = px.bar(
                prob_df, x='Especie', y='Probabilidad',
                title='Probabilidades de Clasificación',
                color='Especie',
                color_discrete_map={'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Mostrar valores ingresados
            st.write("### 🔬 Características Ingresadas")
            input_df = pd.DataFrame({
                'Característica': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                'Valor (cm)': [sepal_length, sepal_width, petal_length, petal_width]
            })
            st.dataframe(input_df)
            
            # Visualizar nueva muestra en gráfico 3D
            st.write("### 🌍 Nueva Muestra en Espacio 3D")
            new_sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            fig_3d_new = create_3d_visualization(X, y, new_sample)
            st.plotly_chart(fig_3d_new, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h4>👨‍🎓 Alexander Gutierrez</h4>
        <p><strong>Universidad de la Costa - Data Mining 2024</strong></p>
        <p>Proyecto de Clasificación de Especies de Iris con Random Forest</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

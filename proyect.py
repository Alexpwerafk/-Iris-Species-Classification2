# =============================================================================
# =============================================================================
# PROYECTO FINAL: SISTEMA AVANZADO DE CLASIFICACIÓN DE ESPECIES DE IRIS
# Universidad de la Costa - Data Mining & Machine Learning
# Desarrollado por: [Tu Nombre]
# =============================================================================

# ||| IMPORTS ORDENADOS POR CATEGORÍA |||
# ─────────────────────────────────────────────────────────────────────────────

# 1. Built-in Python libraries
import os
import base64
from typing import Tuple, Dict, Any

# 2. Core Data Science libraries
import numpy as np
import pandas as pd

# 3. Machine Learning libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 4. Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 5. Streamlit framework
import streamlit as st

# ||| CONFIGURACIÓN DE PÁGINA STREAMLIT - MANEJO DEFENSIVO |||
# ─────────────────────────────────────────────────────────────────────────────
# CRÍTICO: Debe ser el PRIMER comando Streamlit. Usamos try-except para
# manejar recargas en Streamlit Cloud o ejecuciones múltiples que causarían error.
try:
    st.set_page_config(
        page_title="🌺 Iris Classifier Pro - Universidad de la Costa",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded",
        initial_sidebar_width=350,
        menu_items={
            'Get Help': 'https://www.scikitlearn.org',
            'Report a bug': None,
            'About': "Sistema avanzado de clasificación de especies de Iris usando Random Forest optimizado"
        }
    )
except Exception as e:
    # Si ya está configurado o hay un error, continuar sin problema
    # Esto previene el error: "set_page_config() can only be called once"
    pass

# ||| CSS PERSONALIZADO PARA UI/UX PREMIUM |||
# ─────────────────────────────────────────────────────────────────────────────
def load_css():
    """Carga estilos CSS personalizados para una UI profesional"""
    st.markdown("""
    <style>
    /* Header profesional */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Métricas cards */
    .metric-card {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Sliders personalizados */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Botón predict */
    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Tabs personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f0f2f6;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ||| SISTEMA DE CACHÉ AVANZADO |||
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=True)
def load_and_explore_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga y realiza análisis exploratorio inicial del dataset Iris.
    
    Returns:
        tuple: (features_dataframe, target_series)
    """
    # Cargar dataset de sklearn
    iris = load_iris()
    
    # Crear DataFrame con nombres descriptivos
    df_features = pd.DataFrame(
        iris.data, 
        columns=['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
    )
    df_features['Species'] = iris.target_names[iris.target]
    
    # Crear series separadas para features y target
    X = df_features.drop('Species', axis=1)
    y = pd.Series(iris.target_names[iris.target], name='Species')
    
    return X, y

@st.cache_resource(show_spinner="🤖 Entrenando modelo optimizado Random Forest...")
def create_ml_pipeline() -> Tuple[Pipeline, dict]:
    """
    Crea un pipeline completo de ML con preprocesamiento y modelo optimizado.
    
    Returns:
        Tuple[Pipeline, dict]: Pipeline con StandardScaler y Random Forest + parámetros GridSearch
    """
    # Hiperparámetros para GridSearchCV (optimizados para el dataset Iris)
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    # Crear pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    return pipeline, param_grid

@st.cache_resource(show_spinner="⚡ Aplicando GridSearchCV con Cross-Validation...")
def train_and_evaluate_model(pipeline: Pipeline, param_grid: dict, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Entrena el modelo con GridSearchCV y evalúa métricas de rendimiento.
    
    Args:
        pipeline: Pipeline de scikit-learn
        param_grid: Diccionario de hiperparámetros para GridSearchCV
        X: Features
        y: Target
    
    Returns:
        dict: Contiene modelo entrenado, métricas y resultados
    """
    # Dividir datos (80/20) con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # GridSearchCV con validación cruzada de 5 pliegues
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    
    # Entrenar modelo
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo
    best_model = grid_search.best_estimator_
    
    # Predicciones
    y_pred = best_model.predict(X_test)
    
    # Métricas detalladas
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=['setosa', 'versicolor', 'virginica'])
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='accuracy')
    
    return {
        'model': best_model,
        'metrics': metrics,
        'confusion_matrix': cm,
        'cv_scores': cv_scores,
        'X_test': X_test,
        'y_test': y_test,
        'grid_results': pd.DataFrame(grid_search.cv_results_)
    }

@st.cache_data
def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extrae la importancia de características del modelo Random Forest.
    
    Args:
        pipeline: Pipeline entrenado
    
    Returns:
        pd.DataFrame: Importancia de cada característica
    """
    # Obtener el clasificador del pipeline
    rf_model = pipeline.named_steps['classifier']
    
    # Importancia de características
    importance = rf_model.feature_importances_
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    
    # Crear DataFrame ordenado
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    return importance_df

@st.cache_data
def create_3d_visualization(X: pd.DataFrame, y: pd.Series, new_sample: np.ndarray = None) -> go.Figure:
    """
    Crea visualización 3D interactiva usando PCA (captura 95% de varianza).
    
    Args:
        X: Features originales
        y: Labels
        new_sample: Muestra nueva para visualizar (opcional)
    
    Returns:
        plotly.graph_objects.Figure: Gráfico 3D interactivo
    """
    # Aplicar PCA para reducir a 3 componentes (explica 95%+ de varianza en Iris)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    
    # Crear DataFrame para Plotly
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Species'] = y.values
    
    # Crear figura 3D
    fig = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Species',
        title="Visualización 3D con PCA (95% varianza explicada)",
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'},
        color_discrete_map={'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'},
        size_max=10,
        opacity=0.8
    )
    
    # Si hay una nueva muestra, agregarla
    if new_sample is not None:
        new_sample_pca = pca.transform(new_sample.reshape(1, -1))
        fig.add_trace(go.Scatter3d(
            x=[new_sample_pca[0][0]],
            y=[new_sample_pca[0][1]],
            z=[new_sample_pca[0][2]],
            mode='markers',
            marker=dict(size=12, color='yellow', symbol='diamond', line=dict(width=2, color='black')),
            name='Nueva Muestra',
            text=['Nueva Muestra']
        ))
    
    # Actualizar layout
    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} varianza)"
        ),
        legend=dict(x=0, y=0, bgcolor='rgba(255,255,255,0.8)'),
        height=700
    )
    
    return fig

def predict_species(pipeline: Pipeline, sepal_length: float, sepal_width: float, 
                    petal_length: float, petal_width: float) -> Tuple[str, Dict[str, float], np.ndarray]:
    """
    Realiza predicción de especie con probabilidades.
    
    Args:
        pipeline: Modelo entrenado
        sepal_length: Longitud del sépalo
        sepal_width: Ancho del sépalo
        petal_length: Longitud del pétalo
        petal_width: Ancho del pétalo
    
    Returns:
        tuple: (especie_predicha, probabilidades, muestra)
    """
    # Crear array de características
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Realizar predicción
    prediction = pipeline.predict(sample)[0]
    
    # Obtener probabilidades
    probabilities = pipeline.predict_proba(sample)[0]
    prob_dict = dict(zip(pipeline.classes_, probabilities))
    
    return prediction, prob_dict, sample

# ||| LÓGICA PRINCIPAL DE LA APLICACIÓN |||
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """Función principal que ejecuta la aplicación Streamlit"""
    
    # Cargar CSS personalizado
    load_css()
    
    # HEADER PROFESIONAL
    st.markdown("""
    <div class="main-header">
        <h1>🌸 Sistema Avanzado de Clasificación de Especies de Iris</h1>
        <p>Proyecto Final - Universidad de la Costa | Data Mining & Machine Learning</p>
        <p><strong>Modelo:</strong> Random Forest Optimizado con GridSearchCV | <strong>Dataset:</strong> Iris (150 muestras, 3 especies)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR - CONTROLES DE PREDICCIÓN
    # ─────────────────────────────────────────────────────────────────────────
    st.sidebar.title("🔮 Panel de Control")
    st.sidebar.markdown("### 🎚️ Ingrese las Características de la Flor")
    
    # Cargar datos para obtener rangos
    X, y = load_and_explore_data()
    
    # Sliders con valores mín/máx del dataset
    with st.sidebar.form(key='prediction_form'):
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=float(X['Sepal Length (cm)'].min()),
            max_value=float(X['Sepal Length (cm)'].max()),
            value=float(X['Sepal Length (cm)'].mean()),
            step=0.1,
            help="Longitud del sépalo en centímetros"
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=float(X['Sepal Width (cm)'].min()),
            max_value=float(X['Sepal Width (cm)'].max()),
            value=float(X['Sepal Width (cm)'].mean()),
            step=0.1,
            help="Ancho del sépalo en centímetros"
        )
        
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=float(X['Petal Length (cm)'].min()),
            max_value=float(X['Petal Length (cm)'].max()),
            value=float(X['Petal Length (cm)'].mean()),
            step=0.1,
            help="Longitud del pétalo en centímetros"
        )
        
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=float(X['Petal Width (cm)'].min()),
            max_value=float(X['Petal Width (cm)'].max()),
            value=float(X['Petal Width (cm)'].mean()),
            step=0.1,
            help="Ancho del pétalo en centímetros"
        )
        
        # Botón de predicción
        submit_button = st.form_submit_button(label="🚀 Predecir Especie", use_container_width=True)
    
    # TABS PRINCIPALES
    # ─────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔬 Análisis Exploratorio", "🌍 Visualización 3D", "🔮 Predicción"])
    
    # ENTRENAMIENTO DE MODELO (una sola vez)
    # ─────────────────────────────────────────────────────────────────────────
    if 'model_results' not in st.session_state:
        with st.spinner('🤖 Entrenando modelo Random Forest con GridSearchCV...'):
            pipeline, param_grid = create_ml_pipeline()
            results = train_and_evaluate_model(pipeline, param_grid, X, y)
            st.session_state['model_results'] = results
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['pipeline'] = results['model']
    
    # Extraer resultados de session state
    results = st.session_state['model_results']
    pipeline = st.session_state['pipeline']
    
    # TAB 1: DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("## 📊 Dashboard de Rendimiento del Modelo")
        
        # Métricas principales con st.metrics()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🎯 Accuracy",
                value=f"{results['metrics']['accuracy']:.4f}",
                delta=f"{results['metrics']['accuracy']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                label="📍 Precision",
                value=f"{results['metrics']['precision']:.4f}",
                delta="Weighted Avg"
            )
        
        with col3:
            st.metric(
                label="🔄 Recall",
                value=f"{results['metrics']['recall']:.4f}",
                delta="Weighted Avg"
            )
        
        with col4:
            st.metric(
                label="⚖️ F1-Score",
                value=f"{results['metrics']['f1_score']:.4f}",
                delta="Weighted Avg"
            )
        
        # Barras de progreso coloridas para métricas
        st.markdown("### 📈 Visualización de Métricas")
        metrics_cols = st.columns(4)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#4facfe', '#00f2fe', '#43e97b', '#38f9d7']
        emojis = ['🎯', '📍', '🔄', '⚖️']
        
        for i, metric in enumerate(metrics):
            with metrics_cols[i]:
                value = results['metrics'][metric]
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{emojis[i]} {metric.title()}</h3>
                    <div style="background: rgba(255,255,255,0.3); border-radius: 10px; padding: 5px;">
                        <div style="background: linear-gradient(90deg, {colors[i]} 0%, {colors[i]} {value*100}%, transparent {value*100}%); height: 30px; border-radius: 5px; display: flex; align-items: center; padding-left: 10px;">
                            <strong>{value:.4f}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Importancia de características
        st.markdown("### 🏆 Importancia de Características")
        importance_df = get_feature_importance(pipeline)
        
        fig_importance = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Importancia de Características - Random Forest',
            color='Importance',
            color_continuous_scale='viridis',
            text='Importance'
        )
        fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_importance.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Matriz de confusión interactiva con Plotly
        st.markdown("### 🔍 Matriz de Confusión Interactiva")
        cm = results['confusion_matrix']
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='Blues',
            title='Matriz de Confusión - Modelo Random Forest',
            labels=dict(x="Predicción", y="Real", color="Count"),
            x=['Setosa', 'Versicolor', 'Virginica'],
            y=['Setosa', 'Versicolor', 'Virginica']
        )
        fig_cm.update_layout(height=500)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Cross-validation scores
        st.markdown("### 📉 Cross-Validation Scores (10-fold)")
        st.write(f"**Mean CV Accuracy:** {results['cv_scores'].mean():.4f} (±{results['cv_scores'].std():.4f})")
        fig_cv = px.box(y=results['cv_scores'], title='Distribución de Accuracy en CV')
        fig_cv.update_layout(yaxis_title='Accuracy', height=300)
        st.plotly_chart(fig_cv, use_container_width=True)
    
    # TAB 2: ANÁLISIS EXPLORATORIO
    # ─────────────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("## 🔬 Análisis Exploratorio de Datos (EDA)")
        
        # Estadísticas descriptivas
        st.markdown("### 📋 Estadísticas Descriptivas por Especie")
        desc_stats = pd.concat([X, y], axis=1).groupby('Species').describe()
        st.dataframe(desc_stats, use_container_width=True)
        
        # Histogramas con distribuciones por clase
        st.markdown("### 📊 Histogramas por Característica y Especie")
        fig_hist = make_subplots(rows=2, cols=2, subplot_titles=X.columns.tolist())
        
        for idx, feature in enumerate(X.columns):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            for species in y.unique():
                data = X[y == species][feature]
                fig_hist.add_trace(
                    go.Histogram(x=data, name=species, opacity=0.7, nbinsx=15),
                    row=row, col=col
                )
        
        fig_hist.update_layout(height=600, barmode='overlay', showlegend=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Scatter matrix (pairplot) con seaborn
        st.markdown("### 🔗 Scatter Matrix (Pairplot)")
        fig_pairplot = plt.figure(figsize=(12, 10))
        pairplot_data = pd.concat([X, y], axis=1)
        sns.pairplot(pairplot_data, hue='Species', diag_kind='kde', palette='Set2', markers=['o', 's', 'D'])
        st.pyplot(fig_pairplot)
        
        # Violin plots para distribuciones de características
        st.markdown("### 🎻 Violin Plots - Distribución de Características")
        fig_violin = make_subplots(rows=2, cols=2, subplot_titles=X.columns.tolist())
        
        for idx, feature in enumerate(X.columns):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            for species in y.unique():
                data = X[y == species][feature]
                fig_violin.add_trace(
                    go.Violin(y=data, name=species, box_visible=True, meanline_visible=True, opacity=0.6),
                    row=row, col=col
                )
        
        fig_violin.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
    
    # TAB 3: VISUALIZACIÓN 3D
    # ─────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown("## 🌍 Visualización 3D con PCA")
        st.info("ℹ️ **Nota Académica**: Se aplica PCA (Principal Component Analysis) para reducir las 4 dimensiones a 3 componentes principales que explican más del 95% de la varianza total. Esto permite visualizar la estructura del dataset en 3D.")
        
        # Crear y mostrar gráfico 3D
        fig_3d = create_3d_visualization(X, y)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Explicación de varianza
        pca = PCA(n_components=3, random_state=42)
        pca.fit(X)
        st.markdown("### 📊 Varianza Explicada por Componente")
        var_df = pd.DataFrame({
            'Componente Principal': ['PC1', 'PC2', 'PC3'],
            'Varianza Explicada': pca.explained_variance_ratio_,
            'Varianza Acumulada': np.cumsum(pca.explained_variance_ratio_)
        })
        st.dataframe(var_df, use_container_width=True)
        
        st.success(f"✅ **Total de varianza explicada por 3 componentes: {pca.explained_variance_ratio_.sum():.2%}**")
    
    # TAB 4: PREDICCIÓN
    # ─────────────────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("## 🔮 Resultado de Predicción")
        
        if submit_button:
            with st.spinner('🤔 Realizando predicción...'):
                # Realizar predicción
                prediction, probabilities, sample = predict_species(
                    pipeline, sepal_length, sepal_width, petal_length, petal_width
                )
                
                # Mostrar resultado con emoji
                species_emojis = {'setosa': '🌺', 'versicolor': '🌸', 'virginica': '🌼'}
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                    <h2>{species_emojis.get(prediction, '🌿')} Especie Predicha: <strong>{prediction.title()}</strong></h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilidades
                st.markdown("### 📊 Probabilidades por Clase")
                prob_cols = st.columns(3)
                for idx, (species, prob) in enumerate(probabilities.items()):
                    prob_cols[idx].markdown(f"""
                    <div style="background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid #667eea;">
                        <h4>{species.title()}</h4>
                        <h2 style="color: #667eea;">{prob:.2%}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizar punto en gráfico 3D
                st.markdown("### 🗺️ Ubicación en Espacio 3D (PCA)")
                fig_3d_new = create_3d_visualization(X, y, new_sample=sample)
                st.plotly_chart(fig_3d_new, use_container_width=True)
                
                # Características ingresadas
                st.markdown("### 📝 Características Ingresadas")
                input_df = pd.DataFrame([{
                    'Sepal Length': sepal_length,
                    'Sepal Width': sepal_width,
                    'Petal Length': petal_length,
                    'Petal Width': petal_width
                }])
                st.dataframe(input_df, use_container_width=True)
        else:
            st.info("👈 Por favor, ingrese las características en el panel izquierdo y presione 'Predecir Especie' para ver el resultado.")

# ||| PUNTO DE ENTRADA DE LA APLICACIÓN |||
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

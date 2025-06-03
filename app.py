import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import io
import tempfile
import os
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Interface de Classification d'Objets 3D - DGCNN",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Classes ModelNet10
CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 
           'monitor', 'night_stand', 'sofa', 'table', 'toilet']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_graph_feature(x, k=20, idx=None):
    """Construit les features du graphe pour EdgeConv"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature

def knn(x, k):
    """Trouve les k plus proches voisins"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

class DGCNN(nn.Module):
    """Modèle DGCNN pour la classification d'objets 3D"""
    
    def __init__(self, num_classes=10, k=20, dropout=0.5):
        super(DGCNN, self).__init__()
        self.k = k

        # EdgeConv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Global feature extraction
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # EdgeConv 1
        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # EdgeConv 2
        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        # EdgeConv 3
        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        # EdgeConv 4
        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]

        # Concaténation des features
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Global feature
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        # Classification
        x = self.classifier(x)

        return x

@st.cache_resource
def load_model():
    """Charge le modèle DGCNN pré-entraîné"""
    try:
        model = DGCNN(num_classes=10, k=20, dropout=0.5)
        
        # Vérifier si le fichier du modèle existe
        model_path = 'best_dgcnn_model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, True
        else:
            return model, False
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, False

def load_off_file(file_content: bytes) -> np.ndarray:
    """Charge un fichier OFF depuis le contenu en bytes"""
    try:
        content = file_content.decode('utf-8')
        lines = content.strip().split('\n')
        
        # Vérifier l'en-tête OFF
        if lines[0].strip() != 'OFF':
            raise ValueError("Le fichier n'est pas au format OFF")
        
        # Lire le nombre de vertices et faces
        n_vertices, n_faces, _ = map(int, lines[1].split())
        
        # Lire les vertices
        vertices = []
        for i in range(2, 2 + n_vertices):
            vertex = list(map(float, lines[i].split()[:3]))
            vertices.append(vertex)
        
        return np.array(vertices)
    
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier OFF : {e}")

def load_ply_file(file_content: bytes) -> np.ndarray:
    """Charge un fichier PLY simple (format ASCII)"""
    try:
        content = file_content.decode('utf-8')
        lines = content.strip().split('\n')
        
        # Chercher le nombre de vertices
        n_vertices = 0
        header_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                n_vertices = int(line.split()[-1])
            elif line.strip() == 'end_header':
                header_end = i + 1
                break
        
        if n_vertices == 0:
            raise ValueError("Format PLY non supporté ou nombre de vertices non trouvé")
        
        # Lire les vertices
        vertices = []
        for i in range(header_end, header_end + n_vertices):
            if i < len(lines):
                coords = lines[i].split()[:3]
                vertex = [float(coord) for coord in coords]
                vertices.append(vertex)
        
        return np.array(vertices)
    
    except Exception as e:
        raise ValueError(f"Erreur lors de la lecture du fichier PLY : {e}")

def normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalise les points dans une sphère unitaire"""
    # Centrer les points
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Normaliser la taille
    m = np.max(np.sqrt(np.sum(points**2, axis=1)))
    if m > 0:
        points = points / m
    
    return points

def preprocess_points(points: np.ndarray, num_points: int = 1024) -> torch.Tensor:
    """Prétraite les points pour le modèle"""
    # Normalisation
    points = normalize_points(points)
    
    # Échantillonnage
    if len(points) >= num_points:
        indices = np.random.choice(len(points), num_points, replace=False)
        points = points[indices]
    else:
        indices = np.random.choice(len(points), num_points, replace=True)
        points = points[indices]
    
    # Conversion en tensor
    points = torch.FloatTensor(points).unsqueeze(0).transpose(2, 1)
    return points

def predict_object(model: nn.Module, points: torch.Tensor) -> Tuple[str, float, np.ndarray]:
    """Fait une prédiction sur les points 3D"""
    model.eval()
    with torch.no_grad():
        points = points.to(device)
        outputs = model(points)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = CLASSES[predicted.item()]
        confidence_score = confidence.item()
        all_probabilities = probabilities.cpu().numpy().flatten()
        
    return predicted_class, confidence_score, all_probabilities

def create_3d_visualization(points: np.ndarray, title: str = "Objet 3D") -> go.Figure:
    """Crée une visualisation 3D interactive avec Plotly"""
    
    # Normaliser les points pour l'affichage
    points_norm = normalize_points(points.copy())
    
    fig = go.Figure(data=[
        go.Scatter3d(
            x=points_norm[:, 0],
            y=points_norm[:, 1],
            z=points_norm[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=points_norm[:, 2],  # Couleur basée sur la coordonnée Z
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Hauteur Z")
            ),
            hovertemplate="X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='#2c3e50')
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            bgcolor='rgba(240,240,240,0.1)'
        ),
        width=700,
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig

def create_confidence_chart(probabilities: np.ndarray) -> go.Figure:
    """Crée un graphique des probabilités par classe"""
    
    df = pd.DataFrame({
        'Classe': CLASSES,
        'Probabilité': probabilities
    }).sort_values('Probabilité', ascending=True)
    
    # Couleurs différentes pour la classe prédite
    colors = ['#1f77b4' if prob < 0.8 else '#ff7f0e' for prob in df['Probabilité']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Classe'],
            x=df['Probabilité'],
            orientation='h',
            marker=dict(color=colors),
            text=[f'{prob:.1%}' for prob in df['Probabilité']],
            textposition='auto',
            hovertemplate="Classe: %{y}<br>Probabilité: %{x:.1%}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title="Probabilités de Classification par Classe",
        xaxis_title="Probabilité",
        yaxis_title="Classes",
        height=400,
        margin=dict(l=100, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig

def main():
    """Fonction principale de l'interface Streamlit"""
    
    # En-tête principal
    st.markdown('<div class="main-header">🎯 Interface de Classification d\'Objets 3D</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by DGCNN (Dynamic Graph CNN)</div>', 
                unsafe_allow_html=True)
    
    # Chargement du modèle
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Modèle non trouvé !</strong><br>
            Le fichier <code>best_dgcnn_model.pth</code> n'a pas été trouvé dans le répertoire courant.
            Veuillez vous assurer que le modèle entraîné est présent pour utiliser l'interface.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="success-box">
            ✅ <strong>Modèle DGCNN chargé avec succès !</strong><br>
            Le modèle est prêt à classifier vos objets 3D.
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar pour les informations et les paramètres
    with st.sidebar:
        st.markdown("### 📋 Informations du Modèle")
        
        st.markdown(f"""
        <div class="info-box">
            <strong>Architecture :</strong> DGCNN<br>
            <strong>Classes :</strong> {len(CLASSES)}<br>
            <strong>Device :</strong> {device}<br>
            <strong>Points par objet :</strong> 1024
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🏷️ Classes Supportées")
        for i, class_name in enumerate(CLASSES, 1):
            st.markdown(f"{i}. **{class_name}**")
        
        st.markdown("### 📁 Formats Supportés")
        st.markdown("""
        - **OFF** (Object File Format)
        - **PLY** (Polygon File Format - ASCII)
        
        *Note : Les fichiers doivent contenir des nuages de points 3D*
        """)
    
    # Interface principale
    if model_loaded:
        # Section d'upload de fichier
        st.markdown('<div class="sub-header">📂 Chargement du Fichier 3D</div>', 
                    unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier OFF ou PLY",
            type=['off', 'ply'],
            help="Uploadez un fichier contenant un nuage de points 3D"
        )
        
        if uploaded_file is not None:
            try:
                # Lire le contenu du fichier
                file_content = uploaded_file.read()
                
                # Déterminer le type de fichier et charger les points
                if uploaded_file.name.lower().endswith('.off'):
                    points = load_off_file(file_content)
                elif uploaded_file.name.lower().endswith('.ply'):
                    points = load_ply_file(file_content)
                else:
                    st.error("Format de fichier non supporté")
                    return
                
                st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>Fichier chargé avec succès !</strong><br>
                    Nom : {uploaded_file.name}<br>
                    Points : {len(points):,}<br>
                    Dimensions : {points.shape[1]}D
                </div>
                """, unsafe_allow_html=True)
                
                # Créer deux colonnes pour la visualisation et les résultats
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown('<div class="sub-header">🎨 Visualisation 3D</div>', 
                                unsafe_allow_html=True)
                    
                    # Afficher la visualisation 3D
                    fig_3d = create_3d_visualization(points, f"Objet 3D - {uploaded_file.name}")
                    st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Statistiques du nuage de points
                    st.markdown("#### 📊 Statistiques du Nuage de Points")
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.metric("Nombre de Points", f"{len(points):,}")
                        st.metric("Dimension", f"{points.shape[1]}D")
                    
                    with stats_col2:
                        bbox_size = np.ptp(points, axis=0)
                        st.metric("Taille X", f"{bbox_size[0]:.3f}")
                        st.metric("Taille Y", f"{bbox_size[1]:.3f}")
                        if len(bbox_size) > 2:
                            st.metric("Taille Z", f"{bbox_size[2]:.3f}")
                
                with col2:
                    st.markdown('<div class="sub-header">🔮 Classification</div>', 
                                unsafe_allow_html=True)
                    
                    if st.button("🚀 Classifier l'Objet", type="primary"):
                        with st.spinner("Classification en cours..."):
                            try:
                                # Préprocesser les points
                                processed_points = preprocess_points(points, num_points=1024)
                                
                                # Faire la prédiction
                                predicted_class, confidence, probabilities = predict_object(model, processed_points)
                                
                                # Afficher les résultats
                                st.markdown(f"""
                                <div class="prediction-box">
                                    <h2>🎯 Résultat de la Classification</h2>
                                    <h1>{predicted_class.upper()}</h1>
                                    <h3>Confiance : {confidence:.1%}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Graphique des probabilités
                                fig_conf = create_confidence_chart(probabilities)
                                st.plotly_chart(fig_conf, use_container_width=True)
                                
                                # Tableau des résultats détaillés
                                st.markdown("#### 📋 Résultats Détaillés")
                                results_df = pd.DataFrame({
                                    'Classe': CLASSES,
                                    'Probabilité': probabilities,
                                    'Confiance (%)': [f"{p:.1%}" for p in probabilities]
                                }).sort_values('Probabilité', ascending=False)
                                
                                st.dataframe(
                                    results_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Indicateur de qualité de la prédiction
                                if confidence > 0.9:
                                    quality = "Excellente"
                                    color = "#28a745"
                                elif confidence > 0.7:
                                    quality = "Bonne"
                                    color = "#ffc107"
                                elif confidence > 0.5:
                                    quality = "Acceptable"
                                    color = "#fd7e14"
                                else:
                                    quality = "Incertaine"
                                    color = "#dc3545"
                                
                                st.markdown(f"""
                                <div style="background-color: {color}; color: white; padding: 1rem; 
                                           border-radius: 8px; text-align: center; margin: 1rem 0;">
                                    <strong>Qualité de la Prédiction : {quality}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Erreur lors de la classification : {e}")
                    
                    # Section d'aide
                    with st.expander("💡 Conseils pour de meilleurs résultats"):
                        st.markdown("""
                        - **Qualité du fichier** : Assurez-vous que votre fichier contient suffisamment de points (> 500)
                        - **Format** : Les fichiers OFF sont généralement mieux supportés
                        - **Objets supportés** : Le modèle a été entraîné sur ModelNet10 (meubles principalement)
                        - **Orientation** : L'orientation de l'objet peut affecter la classification
                        - **Échelle** : La taille de l'objet est automatiquement normalisée
                        """)
                
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier : {e}")
                st.markdown("""
                **Conseils de dépannage :**
                - Vérifiez que le fichier est au format OFF ou PLY
                - Assurez-vous que le fichier contient des données 3D valides
                - Essayez avec un autre fichier
                """)
    
    # Section d'informations supplémentaires
    st.markdown("---")
    st.markdown('<div class="sub-header">ℹ️ À Propos de DGCNN</div>', unsafe_allow_html=True)
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 Architecture</h4>
            <p>Dynamic Graph CNN utilise des convolutions sur graphes dynamiques pour capturer 
            les relations spatiales entre les points 3D.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Dataset</h4>
            <p>Entraîné sur ModelNet10 avec 10 classes d'objets 3D couramment utilisés 
            dans les environnements intérieurs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col3:
        st.markdown("""
        <div class="metric-card">
            <h4>⚡ Performance</h4>
            <p>Le modèle utilise des EdgeConv pour une classification rapide et précise 
            des nuages de points 3D.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# 🎯 Interface de Classification d'Objets 3D - DGCNN

Une interface web moderne et interactive pour la classification d'objets 3D utilisant le modèle DGCNN (Dynamic Graph CNN).

![Interface Preview](interface_preview.png)

## 🌟 Fonctionnalités

- **🔮 Classification en temps réel** : Classifiez vos objets 3D instantanément
- **🎨 Visualisation 3D interactive** : Explorez vos nuages de points en 3D
- **📊 Analyse détaillée** : Probabilités par classe et métriques de confiance
- **📁 Formats multiples** : Support des fichiers OFF et PLY
- **💻 Interface moderne** : Interface web responsive et intuitive
- **🎯 10 classes d'objets** : Meubles et objets d'intérieur (ModelNet10)

## 🏷️ Classes Supportées

Le modèle peut classifier les objets suivants :
1. **Bathtub** (Baignoire)
2. **Bed** (Lit)
3. **Chair** (Chaise)
4. **Desk** (Bureau)
5. **Dresser** (Commode)
6. **Monitor** (Moniteur)
7. **Night Stand** (Table de nuit)
8. **Sofa** (Canapé)
9. **Table** (Table)
10. **Toilet** (Toilettes)

## 🚀 Installation Rapide

### Option 1 : Lancement automatique (Windows)
```bash
# Double-cliquez sur launch.bat
# ou exécutez en ligne de commande :
launch.bat
```

### Option 2 : Lancement automatique (Python)
```bash
python launch.py
```

### Option 3 : Installation manuelle
```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Créer les fichiers de démonstration
python create_demo.py

# 3. Lancer l'interface
streamlit run app.py
```

## 📋 Prérequis

- **Python 3.8+**
- **Modèle entraîné** : `best_dgcnn_model.pth` (généré par votre notebook)
- **Dépendances** : Listées dans `requirements.txt`

## 📁 Structure du Projet

```
Stage/
├── app.py                    # Interface Streamlit principale
├── launch.py                 # Script de lancement Python
├── launch.bat               # Script de lancement Windows
├── create_demo.py           # Générateur de fichiers de démonstration
├── requirements.txt         # Dépendances Python
├── best_dgcnn_model.pth    # Modèle DGCNN entraîné
├── DGCNN_3d_obj.ipynb      # Notebook d'entraînement
├── demo_files/             # Fichiers d'exemple générés
│   ├── chair_example.off
│   ├── table_example.ply
│   └── ...
└── README.md               # Ce fichier
```

## 🖥️ Utilisation de l'Interface

### 1. Lancement
Après avoir lancé l'application, ouvrez votre navigateur à l'adresse : `http://localhost:8501`

### 2. Chargement d'un fichier
- Cliquez sur **"Browse files"** ou glissez-déposez votre fichier
- Formats supportés : `.off`, `.ply`
- Taille recommandée : 500-5000 points

### 3. Visualisation
- **Vue 3D interactive** : Explorez votre objet en 3D
- **Statistiques** : Nombre de points, dimensions, taille
- **Rotation/Zoom** : Utilisez la souris pour naviguer

### 4. Classification
- Cliquez sur **"🚀 Classifier l'Objet"**
- Obtenez la **classe prédite** et la **confiance**
- Consultez les **probabilités détaillées** par classe

### 5. Analyse des résultats
- **Graphique de confiance** : Barres horizontales des probabilités
- **Tableau détaillé** : Toutes les classes avec leurs scores
- **Indicateur de qualité** : Excellent/Bon/Acceptable/Incertain

## 🔧 Configuration Avancée

### Paramètres du modèle
```python
# Dans app.py, vous pouvez modifier :
DGCNN(num_classes=10, k=20, dropout=0.5)
```

### Préprocessing des points
```python
# Nombre de points utilisés pour la classification
preprocess_points(points, num_points=1024)
```

## 📊 Formats de Fichiers Supportés

### Format OFF (Object File Format)
```
OFF
8 6 0
0.0 0.0 0.0
1.0 0.0 0.0
1.0 1.0 0.0
...
```

### Format PLY (Polygon File Format)
```
ply
format ascii 1.0
element vertex 1000
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 0.0 0.0
...
```

## 🎨 Fichiers de Démonstration

Le script `create_demo.py` génère automatiquement des exemples d'objets 3D :

- **chair_example.off** : Chaise avec dossier et pieds
- **table_example.ply** : Table avec plateau et pieds
- **sphere_example.off** : Sphère (objet rond)

## 🐛 Résolution de Problèmes

### Le modèle n'est pas trouvé
```
⚠️ Modèle non trouvé : best_dgcnn_model.pth
```
**Solution** : Exécutez d'abord votre notebook `DGCNN_3d_obj.ipynb` pour entraîner et sauvegarder le modèle.

### Erreur de format de fichier
```
❌ Erreur lors du chargement du fichier
```
**Solutions** :
- Vérifiez que le fichier est au format OFF ou PLY
- Assurez-vous qu'il contient des coordonnées 3D valides
- Testez avec les fichiers de démonstration

### Problème d'installation
```
❌ Erreur lors de l'installation des dépendances
```
**Solutions** :
```bash
# Mettre à jour pip
python -m pip install --upgrade pip

# Installer manuellement
pip install streamlit torch plotly numpy pandas
```

### Erreur CUDA
```
RuntimeError: CUDA out of memory
```
**Solution** : Le modèle utilisera automatiquement le CPU si CUDA n'est pas disponible.

## 📈 Performance

- **Temps de classification** : < 1 seconde
- **Formats supportés** : OFF, PLY
- **Taille max recommandée** : 10,000 points
- **Précision du modèle** : ~92% sur ModelNet10

## 🔮 Fonctionnalités Futures

- [ ] Support de plus de formats (XYZ, PCD)
- [ ] Classification par lots (multiple fichiers)
- [ ] Export des résultats (JSON, CSV)
- [ ] Visualisation des features internes
- [ ] API REST pour intégration
- [ ] Mode batch processing
- [ ] Support ModelNet40 (40 classes)

## 📝 Notes Techniques

### Architecture DGCNN
- **EdgeConv layers** : 4 couches avec k=20 voisins
- **Global features** : 1024 dimensions
- **Classifier** : 3 couches fully connected
- **Dropout** : 0.5 pour la régularisation

### Préprocessing
- **Normalisation** : Centrage + échelle unitaire
- **Échantillonnage** : 1024 points par objet
- **Augmentation** : Rotation aléatoire (optionnel)

## 🤝 Contribution

Contributions bienvenues ! N'hésitez pas à :
1. Signaler des bugs
2. Proposer des améliorations
3. Ajouter de nouvelles fonctionnalités
4. Améliorer la documentation

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

Développé avec ❤️ pour la classification d'objets 3D

---

**🎯 Bon classement ! Si vous avez des questions, n'hésitez pas à consulter la documentation ou à tester avec les fichiers de démonstration.**

"""
Script de démonstration pour l'interface de classification 3D
Génère des exemples d'objets 3D pour tester l'interface
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_chair_points(num_points=1000):
    """Génère un nuage de points approximant une chaise"""
    points = []
    
    # Assise (rectangle horizontal)
    for _ in range(num_points // 3):
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(0.4, 0.5)
        points.append([x, y, z])
    
    # Dossier (rectangle vertical)
    for _ in range(num_points // 3):
        x = np.random.uniform(-0.5, 0.5)
        y = np.random.uniform(0.4, 0.5)
        z = np.random.uniform(0.5, 1.2)
        points.append([x, y, z])
    
    # Pieds (4 cylindres)
    leg_positions = [(-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4), (0.4, 0.4)]
    for leg_x, leg_y in leg_positions:
        for _ in range(num_points // 12):
            x = leg_x + np.random.uniform(-0.05, 0.05)
            y = leg_y + np.random.uniform(-0.05, 0.05)
            z = np.random.uniform(0.0, 0.4)
            points.append([x, y, z])
    
    return np.array(points)

def generate_table_points(num_points=1000):
    """Génère un nuage de points approximant une table"""
    points = []
    
    # Plateau (rectangle horizontal)
    for _ in range(num_points // 2):
        x = np.random.uniform(-0.8, 0.8)
        y = np.random.uniform(-0.5, 0.5)
        z = np.random.uniform(0.7, 0.8)
        points.append([x, y, z])
    
    # Pieds (4 cylindres)
    leg_positions = [(-0.7, -0.4), (0.7, -0.4), (-0.7, 0.4), (0.7, 0.4)]
    for leg_x, leg_y in leg_positions:
        for _ in range(num_points // 8):
            x = leg_x + np.random.uniform(-0.05, 0.05)
            y = leg_y + np.random.uniform(-0.05, 0.05)
            z = np.random.uniform(0.0, 0.7)
            points.append([x, y, z])
    
    return np.array(points)

def generate_sphere_points(num_points=1000, radius=0.5):
    """Génère un nuage de points sur une sphère (approximation d'un objet rond)"""
    points = []
    
    for _ in range(num_points):
        # Méthode de Marsaglia pour distribution uniforme sur sphère
        u = np.random.uniform(-1, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        
        sqrt_term = np.sqrt(1 - u**2)
        x = radius * sqrt_term * np.cos(theta)
        y = radius * sqrt_term * np.sin(theta)
        z = radius * u
        
        points.append([x, y, z])
    
    return np.array(points)

def save_as_off(points, filename):
    """Sauvegarde les points au format OFF"""
    with open(filename, 'w') as f:
        f.write("OFF\n")
        f.write(f"{len(points)} 0 0\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

def save_as_ply(points, filename):
    """Sauvegarde les points au format PLY"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")

def visualize_examples():
    """Visualise les exemples générés"""
    examples = {
        'Chaise': generate_chair_points(),
        'Table': generate_table_points(),
        'Sphère': generate_sphere_points()
    }
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, (name, points) in enumerate(examples.items(), 1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=points[:, 2], cmap='viridis', s=1, alpha=0.6)
        
        ax.set_title(f'{name}\n({len(points)} points)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Égaliser les échelles
        max_range = 1.0
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
    
    plt.tight_layout()
    plt.savefig('exemples_objets_3d.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_demo_files():
    """Crée des fichiers de démonstration"""
    
    # Créer le dossier de démonstration
    demo_dir = "demo_files"
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    # Générer et sauvegarder les exemples
    examples = {
        'chair_example': generate_chair_points(1500),
        'table_example': generate_table_points(1200),
        'sphere_example': generate_sphere_points(1000),
        'chair_simple': generate_chair_points(800),
        'table_large': generate_table_points(2000)
    }
    
    print("Création des fichiers de démonstration...")
    
    for name, points in examples.items():
        # Sauvegarder en OFF
        off_filename = os.path.join(demo_dir, f"{name}.off")
        save_as_off(points, off_filename)
        
        # Sauvegarder en PLY
        ply_filename = os.path.join(demo_dir, f"{name}.ply")
        save_as_ply(points, ply_filename)
        
        print(f"✅ Créé : {name}.off et {name}.ply ({len(points)} points)")
    
    print(f"\n🎉 Fichiers de démonstration créés dans le dossier '{demo_dir}'")
    print("\nUtilisation :")
    print("1. Lancez l'interface : streamlit run app.py")
    print("2. Uploadez un des fichiers .off ou .ply du dossier demo_files")
    print("3. Testez la classification !")

def main():
    """Fonction principale"""
    print("🚀 Générateur d'exemples pour l'interface de classification 3D")
    print("=" * 60)
    
    # Créer les fichiers de démonstration
    create_demo_files()
    
    # Visualiser les exemples
    print("\n📊 Génération des visualisations...")
    visualize_examples()
    
    print("\n✨ Démonstration prête !")
    print("\nProchaines étapes :")
    print("1. Installez les dépendances : pip install -r requirements.txt")
    print("2. Lancez l'interface : streamlit run app.py")
    print("3. Testez avec les fichiers générés dans demo_files/")

if __name__ == "__main__":
    main()

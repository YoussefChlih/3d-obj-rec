#!/usr/bin/env python3
"""
Script de lancement pour l'interface de classification d'objets 3D
"""

import subprocess
import sys
import os
import webbrowser
import time

def check_python():
    """Vérifie que Python est installé"""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ requis. Version actuelle:", sys.version)
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} détecté")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la vérification de Python: {e}")
        return False

def install_requirements():
    """Installe les dépendances requises"""
    print("📦 Installation des dépendances...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dépendances installées avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation des dépendances: {e}")
        return False

def check_model():
    """Vérifie la présence du modèle"""
    model_path = 'best_dgcnn_model.pth'
    if os.path.exists(model_path):
        print(f"✅ Modèle trouvé: {model_path}")
        return True
    else:
        print(f"⚠️  Modèle non trouvé: {model_path}")
        print("   L'interface fonctionnera en mode démonstration")
        return False

def create_demo_files():
    """Crée les fichiers de démonstration"""
    print("🎨 Création des fichiers de démonstration...")
    try:
        subprocess.check_call([sys.executable, 'create_demo.py'])
        print("✅ Fichiers de démonstration créés")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Erreur lors de la création des fichiers de démo: {e}")
        return False

def launch_streamlit():
    """Lance l'application Streamlit"""
    print("🚀 Lancement de l'interface web...")
    print("📱 L'interface s'ouvrira dans votre navigateur par défaut")
    print("🛑 Pour arrêter l'application, appuyez sur Ctrl+C")
    print("=" * 50)
    
    try:
        # Ouvrir le navigateur après un délai
        def open_browser():
            time.sleep(3)
            webbrowser.open('http://localhost:8501')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Lancer Streamlit
        subprocess.check_call([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
        
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée par l'utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors du lancement de Streamlit: {e}")
        print("💡 Essayez d'installer Streamlit manuellement: pip install streamlit")

def main():
    """Fonction principale"""
    print("=" * 60)
    print("🎯 Interface de Classification d'Objets 3D")
    print("   Powered by DGCNN (Dynamic Graph CNN)")
    print("=" * 60)
    
    # Vérifications préliminaires
    if not check_python():
        sys.exit(1)
    
    # Vérifier les fichiers requis
    required_files = ['app.py', 'requirements.txt', 'create_demo.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        sys.exit(1)
    
    print("✅ Tous les fichiers requis sont présents")
    
    # Installation des dépendances
    if not install_requirements():
        print("❌ Échec de l'installation des dépendances")
        sys.exit(1)
    
    # Vérification du modèle
    check_model()
    
    # Création des fichiers de démonstration
    create_demo_files()
    
    # Lancement de l'interface
    print("\n" + "=" * 50)
    launch_streamlit()

if __name__ == "__main__":
    main()

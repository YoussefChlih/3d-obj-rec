@echo off
echo ========================================
echo Interface de Classification d'Objets 3D
echo Powered by DGCNN
echo ========================================
echo.

:: Vérifier si Python est installé
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Erreur: Python n'est pas installe ou n'est pas dans le PATH
    echo Veuillez installer Python depuis https://python.org
    pause
    exit /b 1
)

:: Vérifier si le modèle existe
if not exist "best_dgcnn_model.pth" (
    echo Attention: Le fichier du modele 'best_dgcnn_model.pth' n'est pas trouve
    echo L'interface fonctionnera en mode demonstration sans classification
    echo.
)

:: Installer les dépendances si nécessaire
echo Installation des dependances...
pip install -r requirements.txt

:: Créer les fichiers de démonstration
echo.
echo Creation des fichiers de demonstration...
python create_demo.py

:: Lancer l'interface
echo.
echo Lancement de l'interface web...
echo L'interface s'ouvrira automatiquement dans votre navigateur
echo Pour arreter l'application, appuyez sur Ctrl+C dans cette fenetre
echo.
echo URL: http://localhost:8501
echo.
streamlit run app.py --server.port 8501 --server.address localhost

echo.
echo Interface fermee.
pause

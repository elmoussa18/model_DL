# Préparation au déploiement :
# 1) allège les modèles .h5 du dossier Projet (supprime l'état de l'optimiseur, inutile pour la prédiction)
# 2) écrit requirements.txt avec les versions installées dans ce .venv (pour que Streamlit Cloud ait les mêmes)
import os
import sys
import glob
import platform
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from tensorflow.keras.models import load_model

DOSSIER = os.path.dirname(os.path.abspath(__file__))
LIMITE_MO = 95   # GitHub refuse les fichiers de plus de 100 Mo

def taille_mo(chemin):
    return os.path.getsize(chemin) / 1024 / 1024

# ------------------------------------------------ 1) alléger les modèles .h5
print('=== Modèles .h5 ===')
modeles = sorted(glob.glob(os.path.join(DOSSIER, 'Projet', '**', '*.h5'), recursive = True))
if not modeles:
    print("Aucun fichier .h5 trouvé dans le dossier 'Projet'.")
for chemin in modeles:
    nom = os.path.relpath(chemin, DOSSIER)
    avant = taille_mo(chemin)
    # chargement sans compilation : l'optimiseur n'est pas conservé
    model = load_model(chemin, compile = False)
    tmp = chemin[:-3] + '.tmp.h5'
    model.save(tmp)
    # vérification : le modèle allégé doit donner les mêmes prédictions
    forme = [1 if d is None else d for d in model.input_shape[1:]]
    x = np.random.rand(1, *forme).astype('float32')
    p_avant = model.predict(x, verbose = 0)
    p_apres = load_model(tmp, compile = False).predict(x, verbose = 0)
    apres = taille_mo(tmp)
    if np.allclose(p_avant, p_apres, atol = 1e-5) and apres < avant:
        os.replace(tmp, chemin)
        print(f'{nom} : {avant:.1f} Mo -> {apres:.1f} Mo (allégé)')
        apres_final = apres
    else:
        os.remove(tmp)
        print(f'{nom} : {avant:.1f} Mo (déjà léger, inchangé)')
        apres_final = avant
    if apres_final > LIMITE_MO:
        print(f'   ATTENTION : {nom} dépasse {LIMITE_MO} Mo, GitHub le refusera (limite 100 Mo). '
              f"Réduisez les couches Dense dans le tuner ou utilisez Git LFS.")

# ------------------------------------------------ 2) requirements.txt avec les versions installées
print('\n=== requirements.txt ===')
paquets = [['streamlit'], ['tensorflow', 'tensorflow-cpu'], ['numpy'], ['scikit-learn'], ['joblib'],
           ['pandas'], ['pillow'], ['opencv-python-headless', 'opencv-python']]
lignes = []
for choix in paquets:
    for nom in choix:
        try:
            v = version(nom)
            # sur le serveur on utilise la version "headless" d'opencv (sans interface graphique)
            nom_req = 'opencv-python-headless' if nom.startswith('opencv-python') else nom
            nom_req = 'tensorflow' if nom == 'tensorflow-cpu' else nom_req
            lignes.append(f'{nom_req}=={v}')
            break
        except PackageNotFoundError:
            continue
    else:
        print(f"Paquet non installé dans ce .venv : {choix[0]} (installez-le avec pip install {choix[0]})")
with open(os.path.join(DOSSIER, 'requirements.txt'), 'w') as f:
    f.write('\n'.join(lignes) + '\n')
print('\n'.join(lignes))
print(f"\nVersion de Python de ce .venv : {platform.python_version()}")
print("-> Dans Streamlit Cloud (Advanced settings), choisissez la même version de Python (x.y).")

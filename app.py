# Application Streamlit - Projet Deep Learning (Notebooks 1 à 5)
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Dossier de l'application (permet de lancer "streamlit run" depuis n'importe où)
DOSSIER_APP = os.path.dirname(os.path.abspath(__file__))
def trouver(nom_fichier, tp):
    # cherche le fichier dans le dossier du TP (Projet/TPx, artifacts/tpx, models), retourne None s'il n'existe pas
    dossiers = [os.path.join(DOSSIER_APP, 'Projet', tp),
                os.path.join(DOSSIER_APP, 'artifacts', tp.lower()),
                os.path.join(DOSSIER_APP, 'models')]
    for d in dossiers:
        chemin = os.path.join(d, nom_fichier)
        if os.path.exists(chemin):
            return chemin
    return None

# Variables catégorielles explicatives (dans l'ordre des colonnes du jeu de données)
FEATURES_CAT = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                'contact', 'month', 'day_of_week', 'poutcome']

# Charger une seule fois le modèle, le scaler, les encodeurs et les valeurs uniques
@st.cache_resource
def charger_tp1():
    chemin_modele = trouver('DNN_model.h5', 'TP1')
    if chemin_modele is None:
        return None
    model = load_model(chemin_modele, compile = False)
    scaler = joblib.load(trouver('scaler.joblib', 'TP1'))
    uniques = joblib.load(trouver('uniques.joblib', 'TP1'))
    encoders = {col: joblib.load(trouver(f'{col}_encoder.joblib', 'TP1')) for col in FEATURES_CAT}
    return model, scaler, uniques, encoders

# ---------------------------------------------------------------- Page TP1
def page_tp1():
    st.title('Notebook 1 - ANN-Bank-Data')
    st.write("Renseignez les informations du client pour prédire s'il va souscrire.")

    ressources = charger_tp1()
    if ressources is None:
        st.error("Fichiers du modèle introuvables. Exécutez le Notebook 1 jusqu'à la sauvegarde "
                 "(dossier 'Projet/TP1' à côté de app.py).")
        return
    model, scaler, uniques, encoders = ressources

    # Formulaire de saisie (Number et Dropdown)
    with st.form('formulaire_tp1'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('age', min_value = 17, max_value = 100, value = 44, step = 1)
            job = st.selectbox('job', uniques['job'])
            marital = st.selectbox('marital', uniques['marital'])
            education = st.selectbox('education', uniques['education'])
            default = st.selectbox('default', uniques['default'])
            housing = st.selectbox('housing', uniques['housing'])
            loan = st.selectbox('loan', uniques['loan'])
            contact = st.selectbox('contact', uniques['contact'])
        with col2:
            month = st.selectbox('month', uniques['month'])
            day_of_week = st.selectbox('day_of_week', uniques['day_of_week'])
            duration = st.number_input('duration (secondes)', min_value = 0, max_value = 5000, value = 210, step = 1)
            campaign = st.number_input('campaign', min_value = 1, max_value = 60, value = 1, step = 1)
            pdays = st.number_input('pdays (999 = jamais contacté)', min_value = 0, max_value = 999, value = 999, step = 1)
            previous = st.number_input('previous', min_value = 0, max_value = 10, value = 0, step = 1)
            poutcome = st.selectbox('poutcome', uniques['poutcome'])
        envoyer = st.form_submit_button('Prédire')

    if envoyer:
        # même prétraitement que lors de l'entraînement : encodeurs puis scaler
        valeurs_cat = {'job': job, 'marital': marital, 'education': education, 'default': default,
                       'housing': housing, 'loan': loan, 'contact': contact, 'month': month,
                       'day_of_week': day_of_week, 'poutcome': poutcome}
        cat = {col: encoders[col].transform([valeurs_cat[col]])[0] for col in FEATURES_CAT}
        # vecteur dans le même ordre que les colonnes du jeu de données
        x_new = np.array([age, cat['job'], cat['marital'], cat['education'], cat['default'],
                          cat['housing'], cat['loan'], cat['contact'], cat['month'], cat['day_of_week'],
                          duration, campaign, pdays, previous, cat['poutcome']]).reshape(1, -1)
        x_new = scaler.transform(x_new)
        proba = float(model.predict(x_new, verbose = 0)[0][0])

        st.subheader('Résultat')
        if proba > 0.5:
            st.success(f'Le client va **souscrire** (probabilité : {proba:.1%})')
        else:
            st.warning(f'Le client ne va **pas souscrire** (probabilité de souscription : {proba:.1%})')
        st.progress(min(max(proba, 0.0), 1.0))

# Charger une seule fois le modèle, le scaler, l'encodeur et les valeurs uniques du TP2
@st.cache_resource
def charger_tp2():
    chemin_modele = trouver('DNN_student_model.h5', 'TP2')
    if chemin_modele is None:
        return None
    model = load_model(chemin_modele, compile = False)
    scaler = joblib.load(trouver('scaler.joblib', 'TP2'))
    encoder = joblib.load(trouver('ExtracurricularActivities_encoder.joblib', 'TP2'))
    uniques = joblib.load(trouver('uniques.joblib', 'TP2'))
    return model, scaler, encoder, uniques

# ---------------------------------------------------------------- Page TP2
def page_tp2():
    st.title('Notebook 2 - ANN-Student Performance')
    st.write("Renseignez les informations de l'étudiant pour prédire son indice de performance (Performance Index).")

    ressources = charger_tp2()
    if ressources is None:
        st.error("Fichiers du modèle introuvables. Exécutez le Notebook 2 jusqu'à la sauvegarde "
                 "(dossier 'Projet/TP2' à côté de app.py).")
        return
    model, scaler, encoder, uniques = ressources

    # Formulaire de saisie (Number et Dropdown)
    with st.form('formulaire_tp2'):
        hours_studied = st.number_input('Hours Studied (heures étudiées)', min_value = 0, max_value = 24, value = 7, step = 1)
        previous_scores = st.number_input('Previous Scores (notes précédentes)', min_value = 0, max_value = 100, value = 99, step = 1)
        extracurricular = st.selectbox('Extracurricular Activities (activités extrascolaires)', uniques['Extracurricular Activities'])
        sleep_hours = st.number_input('Sleep Hours (heures de sommeil)', min_value = 0, max_value = 24, value = 9, step = 1)
        papers = st.number_input('Sample Question Papers Practiced (sujets pratiqués)', min_value = 0, max_value = 20, value = 1, step = 1)
        envoyer = st.form_submit_button('Prédire')

    if envoyer:
        # même prétraitement que lors de l'entraînement : encodeur puis scaler
        extra = encoder.transform([extracurricular])[0]
        # vecteur dans le même ordre que les colonnes du jeu de données
        x_new = np.array([hours_studied, previous_scores, extra, sleep_hours, papers]).reshape(1, -1)
        x_new = scaler.transform(x_new)
        indice = float(model.predict(x_new, verbose = 0)[0][0])

        st.subheader('Résultat')
        st.metric('Indice de performance prédit', f'{indice:.1f}')
        st.progress(min(max(indice / 100, 0.0), 1.0))

# Classes de CIFAR-10 (dans l'ordre utilisé à l'entraînement)
CLASSES_CIFAR10 = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Charger une seule fois le meilleur modèle du TP3
@st.cache_resource
def charger_tp3():
    chemin_modele = trouver('cifar10_best_model.h5', 'TP3')
    if chemin_modele is None:
        return None
    return load_model(chemin_modele, compile = False)

# Prédiction avec le même prétraitement que l'entraînement (RGB, 32x32, division par 255)
def predire_tp3(model, image):
    image = image.convert('RGB')                       # gris/RGBA -> 3 canaux
    image = image.resize((32, 32))                     # redimensionner à 32x32
    image_array = np.array(image) / 255.0              # normaliser
    image_array = image_array.reshape(1, 32, 32, 3)    # ajouter la dimension du batch
    return model.predict(image_array, verbose = 0)[0]  # probabilités des 10 classes

# ---------------------------------------------------------------- Page TP3
def page_tp3():
    st.title('Notebook 3 - CNN-Cifar10')
    st.write('Téléversez une image : le modèle donne la probabilité de chacune des 10 classes '
             '(avion, voiture, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion).')

    model = charger_tp3()
    if model is None:
        st.error("Fichier du modèle introuvable. Exécutez le Notebook 3 jusqu'à la sauvegarde "
                 "(dossier 'Projet/TP3' à côté de app.py).")
        return

    fichier = st.file_uploader('Choisir une image', type = ['png', 'jpg', 'jpeg'])
    if fichier is not None:
        image = Image.open(fichier)
        st.image(image, caption = 'Image téléversée', width = 250)

        probas = predire_tp3(model, image)
        classe = CLASSES_CIFAR10[int(np.argmax(probas))]
        st.subheader('Résultat')
        st.success(f'Classe prédite : **{classe}** ({probas.max():.1%})')

        # probabilités des 10 classes
        tableau = pd.DataFrame({'Classe': CLASSES_CIFAR10, 'Probabilité': probas}).set_index('Classe')
        st.bar_chart(tableau)
        st.dataframe(tableau.sort_values('Probabilité', ascending = False).style.format('{:.2%}'))

# Charger une seule fois un modèle de radiographie (TP5 ou pneumonie)
@st.cache_resource
def charger_modele_radio(dossier_tp, nom_fichier):
    chemin_modele = trouver(nom_fichier, dossier_tp)
    if chemin_modele is None:
        return None
    return load_model(chemin_modele, compile = False)

# Prétraitement identique à celui de l'entraînement (get_data) :
# niveaux de gris, 150x150, division par 255, canal gris répété 3 fois
def pretraiter_radio(image):
    import cv2  # importé ici pour que les autres pages marchent même sans opencv
    img = np.array(image.convert('L'))                  # niveaux de gris
    img = cv2.resize(img, (150, 150))                   # même redimensionnement que get_data
    img = img / 255.0                                   # normaliser
    img = np.repeat(img[..., np.newaxis], 3, axis = -1) # répéter le canal gris 3 fois
    return img.reshape(1, 150, 150, 3).astype('float32')

# Prédiction : retourne la probabilité de la classe positive (1)
def predire_radio(model, image):
    return float(model.predict(pretraiter_radio(image), verbose = 0)[0][0])

# Page commune aux modèles de radiographies (classe 0 = Normal, classe 1 = maladie)
def page_radio(titre, dossier_tp, nom_fichier, maladie):
    st.title(titre)
    st.warning('Outil pédagogique, ne remplace pas un diagnostic médical.')

    model = charger_modele_radio(dossier_tp, nom_fichier)
    if model is None:
        st.error(f"Fichier du modèle introuvable. Exécutez le notebook correspondant jusqu'à la sauvegarde "
                 f"(dossier 'Projet/{dossier_tp}' à côté de app.py).")
        return

    fichier = st.file_uploader('Téléverser une radiographie thoracique', type = ['png', 'jpg', 'jpeg'],
                               key = f'upload_{dossier_tp}')
    if fichier is not None:
        image = Image.open(fichier)
        st.image(image, caption = 'Radiographie téléversée', width = 300)

        p_maladie = predire_radio(model, image)
        probas = {'Normal': 1 - p_maladie, maladie: p_maladie}

        st.subheader('Résultat')
        if p_maladie > 0.5:
            st.error(f'Classe prédite : **{maladie}** ({p_maladie:.1%})')
        else:
            st.success(f'Classe prédite : **Normal** ({1 - p_maladie:.1%})')

        # probabilité de chaque classe
        tableau = pd.DataFrame({'Classe': list(probas.keys()), 'Probabilité': list(probas.values())}).set_index('Classe')
        st.bar_chart(tableau)
        st.dataframe(tableau.style.format('{:.2%}'))
        st.caption('Outil pédagogique, ne remplace pas un diagnostic médical.')

# ---------------------------------------------------------------- Pages TP5 et Pneumonie
def page_tp5():
    page_radio('Notebook 4 - CNN-Tuberculose',
               'TP5', 'tuberculosis_best_model.h5', 'Tuberculosis')

def page_pneumonie():
    page_radio('Notebook 5 - CNN-Pneumonie',
               'Pneumonie', 'pneumonia_best_model.h5', 'Pneumonia')

# ---------------------------------------------------------------- Menu latéral
st.set_page_config(page_title = 'Projet Deep Learning', layout = 'wide')
st.sidebar.title('Modèles')
pages = {'Notebook 1 - ANN-Bank-Data': page_tp1,
         'Notebook 2 - ANN-Student Performance': page_tp2,
         'Notebook 3 - CNN-Cifar10': page_tp3,
         'Notebook 4 - CNN-Tuberculose': page_tp5,
         'Notebook 5 - CNN-Pneumonie': page_pneumonie}
choix = st.sidebar.radio('Choisir un modèle', list(pages.keys()))
pages[choix]()

import streamlit as st
import numpy as np
import cv2
from descriptor import glcm
from scipy.spatial.distance import cityblock, euclidean, chebyshev, canberra
import os
import sqlite3
import hashlib
import face_recognition
import pickle
import time

dataset_directory = "datasets"

# Fonction pour hasher les mots de passe
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Fonction pour créer un nouvel utilisateur
def create_user(username, email, password, face_image_path):
    conn = sqlite3.connect('authentification.db')
    c = conn.cursor()

    # Générer l'encodage facial
    face_encoding = None
    if os.path.exists(face_image_path):
        encoding = detect_face(face_image_path)
        if encoding is not None:
            face_encoding = pickle.dumps(encoding)  # Sérialiser l'encodage

    # Insérer les données de l'utilisateur avec l'encodage facial
    c.execute("""
        INSERT INTO users (username, email, password, face_encoding) 
        VALUES (?, ?, ?, ?)
    """, (username, email, hash_password(password), face_encoding))
    conn.commit()
    conn.close()

# Fonction pour authentifier un utilisateur
def authenticate_user(username, password):
    conn = sqlite3.connect('authentification.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", 
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    return user

# Fonction pour détecter un visage avec dlib
def detect_face(image_path):
    """
    Utilise face_recognition pour extraire les descripteurs faciaux.
    """
    try:
        # Charger l'image
        image = face_recognition.load_image_file(image_path)
        # Extraire les encodings faciaux
        encodings = face_recognition.face_encodings(image)
        if len(encodings) == 0:
            return None
        return encodings[0]  # Retourne le premier encodage (un seul visage)
    except Exception as e:
        return None

# Fonction pour comparer deux encodages faciaux
def compare_faces(reference_encoding, webcam_frame_path):
    webcam_encoding = detect_face(webcam_frame_path)
    if webcam_encoding is None:
        return False, "Aucun visage détecté avec la webcam."

    # Comparer les encodages avec une tolérance
    results = face_recognition.compare_faces([reference_encoding], webcam_encoding, tolerance=0.6)
    if results[0]:
        return True, "Visage reconnu avec succès."
    else:
        return False, "Aucun visage correspondant trouvé."

# Fonction pour charger les signatures et les chemins
def load_signatures_and_paths(signatures):
    numeric_signatures = []
    file_paths = []
    file_count = 0
    for root, dirs, files in os.walk(dataset_directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                if file_count >= len(signatures):
                    break
                path = os.path.join(root, file)
                sig = signatures[file_count]
                numeric_sig = []
                for item in sig:
                    try:
                        numeric_sig.append(float(item))
                    except ValueError:
                        continue
                if numeric_sig:
                    numeric_signatures.append(np.array(numeric_sig))
                    file_paths.append(path)
                file_count += 1
    return numeric_signatures, file_paths

# Charger les signatures pré-calculées
signatures = np.load('signatures.npy', allow_pickle=True)
numeric_signatures, file_paths = load_signatures_and_paths(signatures)

# Définir les fonctions de distance
distance_functions = {
    "Manhattan": cityblock,
    "Euclidean": euclidean,
    "Chebyshev": chebyshev,
    "Canberra": canberra
}

def calculate_descriptor(image_path, descriptor):
    if descriptor == "GLCM":
        return glcm(image_path)
    else:
        raise ValueError("Unknown descriptor")

# Fonction pour afficher l'interface CBIR
def display_cbir_interface():
    st.subheader("Recherche d'images basée sur le contenu")
    uploaded_file = st.file_uploader("Choisissez une image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 0)

        # Enregistrer l'image téléchargée
        upload_path = "uploaded_image.png"
        cv2.imwrite(upload_path, image)

        st.image(image, caption='Image téléchargée.', use_column_width=True)

        # Choisir le descripteur
        descriptor_choice = st.selectbox("Choisissez un descripteur:", ["GLCM"])

        # Choisir la métrique de distance
        distance_choice = st.selectbox("Choisissez une métrique de distance:", ("Manhattan", "Euclidean", "Chebyshev", "Canberra"))

        # Nombre de résultats à afficher
        num_results = st.slider("Nombre de résultats à afficher:", 1, 10, 5)

        if st.button("Lancer la recherche"):
            # Calculer le descripteur pour l'image téléchargée
            uploaded_descriptor = calculate_descriptor(upload_path, descriptor_choice)
            uploaded_descriptor = np.array(uploaded_descriptor, dtype=float).flatten()

            distances = []
            for signature in numeric_signatures:
                dist = distance_functions[distance_choice](uploaded_descriptor, signature.flatten())
                distances.append(dist)

            # Trier par distance et obtenir les meilleurs résultats
            sorted_indices = np.argsort(distances)
            top_indices = sorted_indices[:num_results]

            st.write(f"Top {num_results} images similaires:")

            for idx in top_indices:
                result_image_path = file_paths[idx]
                result_distance = distances[idx]
                if os.path.exists(result_image_path):
                    st.image(result_image_path, caption=f"Distance: {result_distance}", use_column_width=True)
                else:
                    st.write(f"Image non trouvée: {result_image_path} (Distance: {result_distance})")

# Interface principale
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

st.title("Authentification et CBIR")

if not st.session_state["authenticated"]:
    choice = st.selectbox("Choisissez une option", ["Connexion", "Inscription", "Reconnaissance Faciale"])

    if choice == "Inscription":
        st.subheader("Créer un compte")
        username = st.text_input("Nom d'utilisateur")
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        face_image_path = None
        uploaded_file = st.file_uploader("Photo (JPG ou PNG)", type=["jpg", "png"])
        if uploaded_file is not None:
            face_image_path = os.path.join("reference_images", uploaded_file.name)
            with open(face_image_path, "wb") as f:
                f.write(uploaded_file.read())
            st.image(face_image_path, caption="Photo téléchargée.", use_column_width=True)

        if st.button("S'inscrire"):
            if username and email and password and face_image_path:
                create_user(username, email, password, face_image_path)
                st.success("Compte créé avec succès !")
            else:
                st.error("Veuillez remplir tous les champs et télécharger une photo.")

    elif choice == "Connexion":
        st.subheader("Se connecter")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            user = authenticate_user(username, password)
            if user:
                st.success(f"Bienvenue {username} ! Vous êtes connecté.")
                st.session_state["authenticated"] = True
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")

    elif choice == "Reconnaissance Faciale":
        st.subheader("Authentification par Reconnaissance Faciale")
        reference_path = "reference_images/moi.jpg"
        if not os.path.exists(reference_path):
            st.error("L'image de référence est introuvable. Ajoutez une image dans le dossier 'reference_images'.")
        else:
            try:
                st.warning("La webcam s'initialise, patientez quelques secondes avant de capturer une image.")
                time.sleep(5)  # Attendre 5 secondes pour stabiliser le flux vidéo

                # Capture temporaire d'une image avec la webcam
                frame_path = "webcam_frame.jpg"
                if st.button("Capturer une image"):
                    cap = cv2.VideoCapture(0)
                    time.sleep(2)  # Attendre pour stabiliser le flux de la webcam
                    ret, frame = cap.read()
                    if ret:
                        cv2.imwrite(frame_path, frame)
                        st.image(frame, caption="Image capturée avec la webcam", channels="BGR")
                    else:
                        st.error("Erreur lors de la capture de l'image.")
                    cap.release()

                # Comparaison avec l'image de référence
                if os.path.exists(frame_path):
                    ref_encoding = detect_face(reference_path)
                    if ref_encoding is None:
                        st.error("Aucun visage détecté dans l'image de référence.")
                    else:
                        is_match, message = compare_faces(ref_encoding, frame_path)
                        if is_match:
                            st.success(message)
                            st.session_state["authenticated"] = True
                        else:
                            st.error(message)
            except Exception as e:
                st.error(f"Erreur : {e}")
else:
    display_cbir_interface()


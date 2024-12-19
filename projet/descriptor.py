from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np

# Fonction pour calculer les descripteurs GLCM
def glcm(image_path):
    data = imread(image_path, as_gray=True)
    co_matrix = graycomatrix(data, distances=[1], angles=[0], symmetric=True, normed=True)
    diss = graycoprops(co_matrix, 'dissimilarity')[0, 0]
    cont = graycoprops(co_matrix, 'contrast')[0, 0]
    corr = graycoprops(co_matrix, 'correlation')[0, 0]
    ener = graycoprops(co_matrix, 'energy')[0, 0]
    asm = graycoprops(co_matrix, 'ASM')[0, 0]
    homo = graycoprops(co_matrix, 'homogeneity')[0, 0]
    return [diss, cont, corr, ener, asm, homo]

# Fonction pour calculer les descripteurs Haralick
def haralick(image_path):
    data = imread(image_path, as_gray=True)
    lbp = local_binary_pattern(data, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(59), density=True)
    return hist.tolist()

# Fonction pour calculer les descripteurs BiT (simplifié)
def bit_descriptor(image_path):
    data = imread(image_path, as_gray=True)
    resized = np.resize(data, (32, 32))  # Réduction à une taille fixe
    return resized.flatten().tolist()

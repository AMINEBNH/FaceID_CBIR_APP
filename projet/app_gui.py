import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
from descriptor import glcm
from distances import euclidean, manhattan, chebyshev, canberra
import os

# Fonction pour charger une image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        path.set(file_path)
        display_image(file_path, panel)
        calculate_distances(file_path)

# Fonction pour afficher l'image chargée
def display_image(img_path, panel):
    img = Image.open(img_path)
    img = img.resize((250, 250), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

# Fonction pour calculer et afficher les distances avec les images des datasets
def calculate_distances(image_path):
    try:
        feat_glcm_k = glcm(image_path)
        
        min_distances = {
            'euclidean': (float('inf'), None),
            'manhattan': (float('inf'), None),
            'chebyshev': (float('inf'), None),
            'canberra': (float('inf'), None)
        }

        for dataset_folder in os.listdir('datasets'):
            folder_path = os.path.join('datasets', dataset_folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    feat_glcm_dataset = glcm(file_path)
                    
                    euclidean_dist = euclidean(feat_glcm_k, feat_glcm_dataset)
                    manhattan_dist = manhattan(feat_glcm_k, feat_glcm_dataset)
                    chebyshev_dist = chebyshev(feat_glcm_k, feat_glcm_dataset)
                    canberra_dist = canberra(feat_glcm_k, feat_glcm_dataset)

                    if euclidean_dist < min_distances['euclidean'][0]:
                        min_distances['euclidean'] = (euclidean_dist, file_path)
                    if manhattan_dist < min_distances['manhattan'][0]:
                        min_distances['manhattan'] = (manhattan_dist, file_path)
                    if chebyshev_dist < min_distances['chebyshev'][0]:
                        min_distances['chebyshev'] = (chebyshev_dist, file_path)
                    if canberra_dist < min_distances['canberra'][0]:
                        min_distances['canberra'] = (canberra_dist, file_path)

        result_text.set(f'''Most similar images:
        Euclidean: {min_distances['euclidean'][1]} (distance={min_distances['euclidean'][0]})
        Manhattan: {min_distances['manhattan'][1]} (distance={min_distances['manhattan'][0]})
        Chebyshev: {min_distances['chebyshev'][1]} (distance={min_distances['chebyshev'][0]})
        Canberra: {min_distances['canberra'][1]} (distance={min_distances['canberra'][0]})''')

        display_image(min_distances['euclidean'][1], panel_euclidean)
        display_image(min_distances['manhattan'][1], panel_manhattan)
        display_image(min_distances['chebyshev'][1], panel_chebyshev)
        display_image(min_distances['canberra'][1], panel_canberra)
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Configuration de l'interface Tkinter
root = tk.Tk()
root.title("Image Similarity")

path = tk.StringVar()

frame = tk.Frame(root)
frame.pack(pady=20)

btn = tk.Button(frame, text="Load Image", command=load_image)
btn.pack(side=tk.LEFT, padx=10)

panel = tk.Label(frame)
panel.pack(side=tk.LEFT)

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, justify=tk.LEFT, font=("Courier", 12))
result_label.pack(pady=20)

frame_results = tk.Frame(root)
frame_results.pack(pady=20)

panel_euclidean = tk.Label(frame_results)
panel_euclidean.pack(side=tk.LEFT, padx=10)
panel_manhattan = tk.Label(frame_results)
panel_manhattan.pack(side=tk.LEFT, padx=10)
panel_chebyshev = tk.Label(frame_results)
panel_chebyshev.pack(side=tk.LEFT, padx=10)
panel_canberra = tk.Label(frame_results)
panel_canberra.pack(side=tk.LEFT, padx=10)

root.mainloop()

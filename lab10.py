#P2

import face_recognition
import random
import numpy as np
import matplotlib.pyplot as plt
import os


# Ruta a la carpeta con las imágenes de rostros
folder_path = "C:/Users/romi1/BD2/lab10/BD2-Lab10_2/lfw"

print(folder_path)

# Obtener la lista de archivos de imagen en la carpeta
file_list = os.listdir(folder_path)

# Seleccionar N pares de rostros aleatorios
N = 5000
print("Tamaño de la lista: ", len(file_list))
random_pairs = random.sample(range(len(file_list)), N)

pairs = [(file_list[i], file_list[j]) for i, j in zip(random_pairs[0::2], random_pairs[1::2])]

distances = []

for pair in pairs:
    image1_path = os.path.join(folder_path, pair[0])
    image2_path = os.path.join(folder_path, pair[1])

    image1 = face_recognition.load_image_file(image1_path)
    image2 = face_recognition.load_image_file(image2_path)

    face_encoding1 = face_recognition.face_encodings(image1)[0]
    face_encoding2 = face_recognition.face_encodings(image2)[0]

    distance = np.linalg.norm(face_encoding1 - face_encoding2)
    distances.append(distance)

# Mostrar histograma de distribución de distancias
plt.hist(distances, bins=50)
plt.xlabel('Distancia')
plt.ylabel('Frecuencia')
plt.title('Histograma de distancias')
plt.show()
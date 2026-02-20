

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_images(folder):
    """Carga imágenes izquierda y derecha desde una carpeta."""
    path_left = os.path.join(folder, "izquierda.jpeg")
    path_right = os.path.join(folder, "derecha.jpeg")
    if not os.path.exists(path_left) or not os.path.exists(path_right):
        raise FileNotFoundError(f"No se encontraron las imágenes en {folder}")
    
    img_left = cv2.imread(path_left)
    img_right = cv2.imread(path_right)
    if img_left is None or img_right is None:
        raise ValueError("Error al cargar imágenes.")
    
    img_left_rgb = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right_rgb = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    return img_left_rgb, img_right_rgb, gray_left, gray_right

def detect_and_match(gray_left, gray_right):
    """Detecta keypoints y descriptores con SIFT (o ORB) y los empareja."""
    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_left, None)
        kp2, des2 = sift.detectAndCompute(gray_right, None)
    except AttributeError:
        print("SIFT no disponible, usando ORB.")
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray_left, None)
        kp2, des2 = orb.detectAndCompute(gray_right, None)
    
    if des1 is None or des2 is None:
        raise RuntimeError("No se encontraron descriptores.")
    
    if des1.dtype == np.float32:  # SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    else:  # ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"Matches encontrados: {len(good_matches)}")
    return kp1, kp2, good_matches

def compute_homography(kp1, kp2, matches):
    """Calcula la homografía usando RANSAC."""
    if len(matches) < 4:
        raise RuntimeError("No hay suficientes matches.")
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("No se pudo encontrar homografía.")
    return H, mask

def warp_and_stitch(img_left, img_right, H):
    """Aplica homografía y combina en panorama."""
    h_left, w_left = img_left.shape[:2]
    h_right, w_right = img_right.shape[:2]
    
    corners_right = np.float32([[0, 0], [w_right, 0], [w_right, h_right], [0, h_right]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_right, H)
    
    all_corners = np.concatenate((np.float32([[0, 0], [w_left, 0], [w_left, h_left], [0, h_left]]).reshape(-1, 1, 2),
                                  transformed_corners), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H_translated = translation @ H
    panorama_size = (x_max - x_min, y_max - y_min)
    
    warped_right = cv2.warpPerspective(img_right, H_translated, panorama_size)
    warped_left = np.zeros_like(warped_right)
    warped_left[-y_min: -y_min + h_left, -x_min: -x_min + w_left] = img_left
    
    # Combinación simple (promedio en solapamiento)
    mask_left = (warped_left > 0).astype(np.float32)
    mask_right = (warped_right > 0).astype(np.float32)
    denominator = mask_left + mask_right
    denominator[denominator == 0] = 1
    panorama = ((warped_left.astype(np.float32) + warped_right.astype(np.float32)) / denominator).astype(np.uint8)
    
    return panorama, warped_left, warped_right

def create_anaglyph(base_img, warped_img):
    """Crea anaglifo: rojo (base), cian (warped)."""
    if base_img.shape != warped_img.shape:
        raise ValueError("Las imágenes deben tener el mismo tamaño.")
    b_base, g_base, r_base = cv2.split(base_img)
    b_warp, g_warp, r_warp = cv2.split(warped_img)
    anaglyph = np.zeros_like(base_img)
    anaglyph[:, :, 0] = b_warp   # azul
    anaglyph[:, :, 1] = g_warp   # verde
    anaglyph[:, :, 2] = r_base   # rojo
    return anaglyph

def process_set(set_folder, output_folder):
    """Procesa un conjunto de imágenes y guarda resultados."""
    print(f"\nProcesando {set_folder}...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Cargar imágenes
    img_left, img_right, gray_left, gray_right = load_images(set_folder)
    
    # Detectar matches
    kp1, kp2, matches = detect_and_match(gray_left, gray_right)
    
    # Calcular homografía
    H, _ = compute_homography(kp1, kp2, matches)
    print("Homografía calculada.")
    
    # Warpear y crear panorama
    panorama, warped_left, warped_right = warp_and_stitch(img_left, img_right, H)
    
    # Guardar panorama
    panorama_bgr = cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, "panorama.jpg"), panorama_bgr)
    print("Panorama guardado.")
    
    # Crear y guardar anaglifo
    anaglyph = create_anaglyph(warped_left, warped_right)
    anaglyph_bgr = cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, "anaglyph.jpg"), anaglyph_bgr)
    print("Anaglifo guardado.")
    
    # Opcional: mostrar las imágenes
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(panorama)
    plt.title("Panorama")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(anaglyph)
    plt.title("Anaglifo")
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, "preview.png"))
    plt.show()
    
    print(f"Resultados guardados en {output_folder}")

def main():
    # Definir las carpetas de entrada y salida
    sets = [
        ("set1", "output_set1"),
        ("set2", "output_set2")
    ]
    
    for input_folder, output_folder in sets:
        try:
            process_set(input_folder, output_folder)
        except Exception as e:
            print(f"Error procesando {input_folder}: {e}")

if __name__ == "__main__":
    main()
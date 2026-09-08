import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    img = cv2.resize(img, (128, 128))
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    
    mean_intensity = np.mean(img)
    edge_density = np.sum(edges > 0) / (128 * 128)
    
    return [mean_intensity, edge_density]

def main():
    print("Starting Visual Pattern Analyzer...")
    
    data_dir = "./data/patterns"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
        print("Please add some test images (.jpg or .png) to this folder and run again.")
        return

    features = []
    file_names = []
    
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(data_dir, filename)
            feat = extract_features(filepath)
            if feat is not None:
                features.append(feat)
                file_names.append(filename)
                
    if len(features) < 2:
        print("Not enough images found. Please add at least 2 images to the folder.")
        return

    print(f"Processed {len(features)} images. Extracting features...")
    
    X = np.array(features)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    print("-" * 30)
    print("CLASSIFICATION RESULTS:")
    print("-" * 30)
    for file, label in zip(file_names, labels):
        print(f"Image: {file}  -->  Pattern Style: Type {label}")

if __name__ == "__main__":
    main()

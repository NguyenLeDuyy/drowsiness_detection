import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Root directory to search for images
BASE_DIR = "dataset/test"  # Change to your image folder

# Load the trained model
try:
    model = load_model("drowsiness_eye_cnn_model.h5")
    print("Successfully loaded model: drowsiness_cnn_model.h5!")
except:
    try:
        model = load_model("best_model.keras")
        print("Successfully loaded model: best_model.keras!")
    except:
        print("Error: Model not found!")
        exit()

# Declare class names
class_names = {0: 'Closed', 1: 'Open'}

def find_all_images(base_dir):
    """Find only .jpg files in the Open and Closed directories"""
    image_paths = []
    valid_dirs = ['Open', 'Closed']
    
    for root, dirs, files in os.walk(base_dir):
        current_dir = os.path.basename(root)
        if current_dir in valid_dirs:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images to analyze in Open/Closed directories")
    return image_paths

# Load saved normalization stats (same as train.py)
try:
    norm_mean = np.load("norm_mean.npy")
    norm_var = np.load("norm_var.npy")
    norm_std = np.sqrt(norm_var + 1e-8)
    print("Loaded normalization stats for test:", norm_mean.shape)
except Exception as e:
    print("Warning: could not load normalization stats, falling back to /255.0", e)
    norm_mean = None
    norm_std = None

def preprocess_frame(frame):
     # convert & resize
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = cv2.resize(frame, (224, 224)).astype(np.float32)

    # apply saved normalization or fallback
    if norm_mean is not None and norm_std is not None:
        # norm_mean/norm_std shape should be (3,)
        frame_array = (frame_array - norm_mean) / norm_std
    else:
        frame_array = frame_array / 255.0

    # Defensive: ensure we have shape (224,224,3)
    # If shape is (1,224,224,3) or (224,224,3,1) etc. -> squeeze
    if frame_array.ndim > 3:
        frame_array = np.squeeze(frame_array)
    # Final expand to batch dimension -> (1,224,224,3)
    return np.expand_dims(frame_array, axis=0)

def get_prediction(preprocessed_frame):
    # ensure numpy array and correct dtype
    arr = np.asarray(preprocessed_frame, dtype=np.float32)

    # Defensive: if there is an extra singleton dimension (e.g. (1,1,224,224,3)), remove it
    while arr.ndim > 4:
        # remove first axis of size 1
        arr = np.squeeze(arr, axis=1)

    # If shape is (1,224,224,3) it's OK. If not, try to reshape if possible (will raise if incompatible)
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    if arr.shape[-3:] != (224, 224, 3):
        raise ValueError(f"Unexpected input image shape before predict: {arr.shape}")

    prediction = model.predict(arr, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = class_names[class_idx]
    return class_name, confidence, class_idx, prediction[0]

def is_drowsy(class_idx):
    # Determine drowsiness state: closed eyes (0)
    return class_idx == 0

def main():
    # Find all images
    image_paths = find_all_images(BASE_DIR)
    
    if not image_paths:
        print("No images found to analyze")
        return
    
    # Prepare to save results
    results = []
    start_time = time.time()
    
    # Determine the number of images to process
    total_images = len(image_paths)
    processed_images = 0
    
    # Process each image
    for img_path in image_paths:
        try:
            # Read image
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Error: Cannot read image {img_path}")
                continue
                
            # Preprocess
            processed_frame = preprocess_frame(frame)
            
            # Predict
            class_name, confidence, class_idx, all_probs = get_prediction(processed_frame)
            
            # Determine drowsiness state
            drowsy = is_drowsy(class_idx)
            
            # Get parent directory info to compare with ground truth label
            parent_dir = os.path.basename(os.path.dirname(img_path))
            
            # Save result
            results.append({
                'image_path': img_path,
                'predicted_class': class_name,
                'predicted_idx': class_idx,
                'confidence': confidence,
                'is_drowsy': drowsy,
                'parent_dir': parent_dir,
                'probabilities': all_probs,
                'is_correct': parent_dir.lower() == class_name.lower()  # Mark correct/incorrect prediction
            })
            
            # Update progress
            processed_images += 1
            if processed_images % 10 == 0 or processed_images == total_images:
                elapsed_time = time.time() - start_time
                images_per_sec = processed_images / elapsed_time
                print(f"Processed {processed_images}/{total_images} images ({images_per_sec:.1f} images/sec)")
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Results statistics
    print("\n===== RESULTS STATISTICS =====")
    print(f"Total images analyzed: {len(df)}")
    
    # Statistics by predicted class
    class_counts = df['predicted_class'].value_counts()
    print("\nPredicted class distribution:")
    for cls, count in class_counts.items():
        percentage = count / len(df) * 100
        print(f"  {cls}: {count} ({percentage:.1f}%)")
    
    # Drowsiness state statistics
    drowsy_count = df['is_drowsy'].sum()
    alert_count = len(df) - drowsy_count
    print("\nDrowsiness state:")
    print(f"  Drowsy: {drowsy_count} ({drowsy_count/len(df)*100:.1f}%)")
    print(f"  Alert: {alert_count} ({alert_count/len(df)*100:.1f}%)")
    
    # Average confidence by class
    print("\nAverage confidence by class:")
    for cls in class_names.values():
        avg_conf = df[df['predicted_class'] == cls]['confidence'].mean()
        if not pd.isna(avg_conf):  # Check if there is data for this class
            print(f"  {cls}: {avg_conf:.4f}")
    
    # Compare with parent directory (assume parent dir is ground truth label)
    correct_predictions = df['is_correct'].sum()
    accuracy = correct_predictions / len(df) if len(df) > 0 else 0
    print(f"\nOverall accuracy: {accuracy:.4f} ({correct_predictions}/{len(df)})")
    
    # NEW: Detailed accuracy statistics by directory (ground truth label)
    print("\n===== DETAILED STATISTICS BY DIRECTORY =====")
    
    # List all unique directories
    unique_dirs = df['parent_dir'].unique()
    
    # Statistics for each directory
    for dir_name in unique_dirs:
        dir_images = df[df['parent_dir'] == dir_name]
        total = len(dir_images)
        correct = dir_images['is_correct'].sum()
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nDirectory '{dir_name}' (Ground truth label):")
        print(f"  Total images: {total}")
        print(f"  Correct predictions: {correct} ({accuracy*100:.1f}%)")
        print(f"  Incorrect predictions: {total-correct} ({(1-accuracy)*100:.1f}%)")
        
        # Detailed analysis of incorrect predictions
        incorrect_predictions = dir_images[dir_images['is_correct'] == False]
        if len(incorrect_predictions) > 0:
            incorrect_class_counts = incorrect_predictions['predicted_class'].value_counts()
            print("  Incorrect prediction distribution:")
            for cls, count in incorrect_class_counts.items():
                percentage = count / len(incorrect_predictions) * 100
                print(f"    - Misclassified as '{cls}': {count} ({percentage:.1f}%)")
    
    # NEW: Create confusion matrix
    print("\n===== CONFUSION MATRIX =====")
    
    # Prepare data for confusion matrix
    y_true = df['parent_dir'].values
    y_pred = df['predicted_class'].values
    
    # List of all classes
    classes = sorted(set(y_true) | set(y_pred))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save confusion matrix
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix to file: confusion_matrix.png")
    plt.show()
    
    # NEW: Save list of incorrect predictions
    incorrect_df = df[df['is_correct'] == False].sort_values('parent_dir')
    incorrect_file = "incorrect_predictions.csv"
    incorrect_df.to_csv(incorrect_file, index=False)
    print(f"Saved list of {len(incorrect_df)} misclassified images to file: {incorrect_file}")
    
    # Save detailed statistics to file
    stats_file = "prediction_stats.csv"
    df.to_csv(stats_file, index=False)
    print(f"Saved detailed statistics to file: {stats_file}")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    df['predicted_class'].value_counts().plot(kind='bar')
    plt.title('Predicted Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('class_distribution.png')
    print("Saved class distribution plot to file: class_distribution.png")
    plt.show()
    
    # Plot correct/incorrect ratio
    plt.figure(figsize=(8, 8))
    correct_count = df['is_correct'].sum()
    incorrect_count = len(df) - correct_count
    plt.pie([correct_count, incorrect_count], labels=['Correct', 'Incorrect'], 
            autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.axis('equal')
    plt.title('Correct/Incorrect Prediction Ratio')
    plt.savefig('accuracy_pie.png')
    print("Saved correct/incorrect ratio plot to file: accuracy_pie.png")
    plt.show()

if __name__ == "__main__":
    main()
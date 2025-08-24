import os
import math
import pandas as pd
import cv2
import matplotlib.pyplot as plt

CSV_PATH = "incorrect_predictions.csv"  # chỉnh nếu cần

def load_csv(path):
    df = pd.read_csv(path)
    print("Found", len(df), "incorrect predictions\n")
    print(df[['image_path','parent_dir','predicted_class','confidence']].to_string(index=False))
    return df

def show_images(df, max_images=12, cols=4):
    rows = math.ceil(min(len(df), max_images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten()
    for i, (_, row) in enumerate(df.head(max_images).iterrows()):
        img_path = os.path.normpath(row['image_path'])
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.getcwd(), img_path)
        img = cv2.imread(img_path)
        if img is None:
            axes[i].text(0.5,0.5, f"Cannot load\n{img_path}", ha='center')
            axes[i].axis('off')
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        title = f"True: {row['parent_dir']}\nPred: {row['predicted_class']} ({row['confidence']:.2f})"
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    # turn off remaining axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    out_file = "incorrect_samples.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nSaved visualization to {out_file}")
    plt.show()

if __name__ == "__main__":
    df = load_csv(CSV_PATH)
    show_images(df, max_images=12, cols=4)
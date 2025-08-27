# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
import os
import sys
import torch
import joblib
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from torchvision import models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== CONFIG =====================
resize_dim = 224
use_gpu = torch.cuda.is_available()

# ---- Paths (adjust if needed) ----
DATA_FOLDER = "./segmented_outputs_combined/test"   # test split with *_foreground.png
MODELS_DIR = "best_model"                          # where combined.py saved models
RESULTS_DIR = "best_model_results"                         # where we'll save plots
os.makedirs(RESULTS_DIR, exist_ok=True)

# Label mapping
classes_map = {"benign": 0, "malignant": 1}

# ===================== LOAD MODELS =====================
print("Loading saved models...")
rf_path = os.path.join(MODELS_DIR, "cnn_a_rf.pkl")
cnn_b_path = os.path.join(MODELS_DIR, "cnn_b.pth")

# Initialize models as None
rf = None
cnn_b = None
cnn_feat = None

# Load CNN-A (RF)
try:
    rf = joblib.load(rf_path)
    print("CNN-A (RF) model loaded.")
    # ===================== FEATURE EXTRACTOR (CNN-A) =====================
    cnn_feat = models.resnet18(pretrained=True)
    cnn_feat = torch.nn.Sequential(*list(cnn_feat.children())[:-1])
    cnn_feat.eval()
    if use_gpu:
        cnn_feat = cnn_feat.cuda()
except FileNotFoundError:
    print(f"WARNING: CNN-A model not found at '{rf_path}'. Skipping its evaluation.")

# Load CNN-B
try:
    cnn_b = models.resnet18(pretrained=False)
    cnn_b.fc = torch.nn.Linear(cnn_b.fc.in_features, 2)
    cnn_b.load_state_dict(torch.load(cnn_b_path, map_location='cuda' if use_gpu else 'cpu'))
    cnn_b.eval()
    if use_gpu:
        cnn_b = cnn_b.cuda()
    print("CNN-B model loaded.\n")
except FileNotFoundError:
    cnn_b = None # Ensure it's None if loading fails
    print(f"WARNING: CNN-B model not found at '{cnn_b_path}'. Skipping its evaluation.\n")

if rf is None and cnn_b is None:
    print("No models found. Exiting.")
    sys.exit(1)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize_dim, resize_dim)),
    transforms.ToTensor()
])

def extract_cnn_a_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = transform(img).unsqueeze(0)
    if use_gpu:
        t = t.cuda()
    with torch.no_grad():
        f = cnn_feat(t).view(1, -1).cpu().numpy().flatten()
    return f

def predict_cnn_b(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = transform(img).unsqueeze(0)
    if use_gpu:
        t = t.cuda()
    with torch.no_grad():
        out = cnn_b(t)
        pred = torch.argmax(out, dim=1).item()
    return pred

# ===================== PLOTTING HELPERS =====================
plt.style.use('default')
sns.set_palette("husl")

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Malignant'],
                yticklabels=['Benign','Malignant'], ax=ax,
                cbar_kws={'label':'Count'})
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    out_fp = os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(out_fp, dpi=300, bbox_inches='tight')
    plt.close(fig)

def metrics_from_report(report_dict):
    out = {}
    out["Accuracy"] = report_dict.get("accuracy", float("nan"))
    # Use metrics for the "Malignant" class as the primary score
    malignant_metrics = report_dict.get("Malignant", {})
    out["Precision"] = malignant_metrics.get("precision", float("nan"))
    out["Recall"]    = malignant_metrics.get("recall", float("nan"))
    out["F1-Score"]  = malignant_metrics.get("f1-score", float("nan"))
    return out

def plot_radar_chart(df):
    # Radar chart across models
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, mdl in enumerate(df['Model']):
        vals = [df.loc[i, m] for m in metrics] + [df.loc[i, metrics[0]]]
        ax.plot(angles, vals, 'o-', label=mdl, color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.1, color=colors[i % len(colors)])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1); ax.set_title('Model Performance Radar', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fp = os.path.join(RESULTS_DIR, 'model_performance_radar.png')
    plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.close()

def plot_summary_table(df):
    # Comparison table image
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    table_data = [[mdl] + [f"{df.loc[i, col]:.3f}" for col in df.columns if col != 'Model']
                  for i, mdl in enumerate(df['Model'])]
    headers = ['Model'] + [c for c in df.columns if c != 'Model']
    tbl = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.5)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor('#4CAF50')
        tbl[(0, j)].set_text_props(weight='bold', color='white')
    ax.set_title('Model Performance Summary', pad=20)
    fp = os.path.join(RESULTS_DIR, 'model_performance_table.png')
    plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.close()

# ===================== RUN PREDICTIONS (CNN-A & CNN-B) =====================
print("Running predictions on test images...\n")
results = []
for cls in ['benign', 'malignant']:
    folder = os.path.join(DATA_FOLDER, cls)
    # Be extension-agnostic to avoid case issues
    files = glob(os.path.join(folder, "*_foreground.*"))
    for path in tqdm(files, desc=f"{cls} images"):
        label = classes_map[cls]
        base = os.path.basename(path)

        # CNN-A
        pred_a = None
        if rf and cnn_feat:
            feat = extract_cnn_a_feature(path)
            if feat is None:
                continue
            pred_a = rf.predict([feat])[0]

        # CNN-B
        pred_b = None
        if cnn_b:
            pred_b = predict_cnn_b(path)
            if pred_b is None:
                continue

        results.append({
            "filename": base,
            "true_label": label,
            "cnn_a_pred": pred_a,
            "cnn_b_pred": pred_b
        })

if not results:
    print("No test images found. Expected files like './segmented_outputs_combined/test/<class>/*_foreground.png'.")
    sys.exit(1)

# ===================== REPORTS & CONFUSION MATRICES =====================
y_true  = [r['true_label']  for r in results]
rows = [] # For comparison dataframe

# --- CNN-A Evaluation ---
if rf:
    y_pred_a = [r['cnn_a_pred'] for r in results]
    report_a = classification_report(y_true, y_pred_a, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0)
    print("=== CNN-A (Feature + RF) ===")
    print(classification_report(y_true, y_pred_a, target_names=["Benign", "Malignant"], zero_division=0))
    plot_confusion_matrix(y_true, y_pred_a, "CNN-A")
    row = {"Model": "CNN-A (Feature+RF)"}
    row.update(metrics_from_report(report_a))
    rows.append(row)

# --- CNN-B Evaluation ---
if cnn_b:
    y_pred_b = [r['cnn_b_pred'] for r in results]
    report_b = classification_report(y_true, y_pred_b, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0)
    print("=== CNN-B (Direct Classifier) ===")
    print(classification_report(y_true, y_pred_b, target_names=["Benign", "Malignant"], zero_division=0))
    plot_confusion_matrix(y_true, y_pred_b, "CNN-B")
    row = {"Model": "CNN-B (Foreground)"}
    row.update(metrics_from_report(report_b))
    rows.append(row)


# ===================== COMPARISON DF + RADAR + TABLE =====================
if not rows:
    print("\nNo models were evaluated. Cannot generate comparison plots.")
    sys.exit(0)

df_comp = pd.DataFrame(rows, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
])

# Always generate the summary table image if there's at least one model
plot_summary_table(df_comp)

# Only generate comparison plots if there are multiple models
if len(df_comp) > 1:
    # Save bar-style comparison for Accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(df_comp['Model'], df_comp['Accuracy'], alpha=0.7)
    ax.set_title('Accuracy Comparison'); ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
    for b, v in zip(bars, df_comp['Accuracy']):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha='center')
    plt.tight_layout()
    fp = os.path.join(RESULTS_DIR, 'accuracy_model_comparison.png')
    plt.savefig(fp, dpi=300, bbox_inches='tight'); plt.close()

    # Radar chart
    plot_radar_chart(df_comp)

# Print comparison table to console
print("\nFINAL RESULTS SUMMARY:\n", df_comp.to_string(index=False, float_format='%.3f'))
if len(df_comp) > 0:
    best_idx = df_comp['Accuracy'].idxmax()
    print(f"\nBest model: {df_comp.loc[best_idx, 'Model']} (Accuracy: {df_comp.loc[best_idx, 'Accuracy']:.3f})")

# ===================== SAVE PER-IMAGE PREDICTIONS (AND PRINT) =====================
demo_df = pd.DataFrame(results)
# Reorder and filter columns based on which models were loaded
final_cols = ["filename", "true_label"]
if rf:
    final_cols.append("cnn_a_pred")
if cnn_b:
    final_cols.append("cnn_b_pred")
demo_df = demo_df[final_cols]

excel_out = os.path.join(RESULTS_DIR, "model_predictions.xlsx")
demo_df.to_excel(excel_out, index=False)
print(f"\n Saved per-image predictions to: {excel_out}")

print("\nSAMPLE OF PER-IMAGE PREDICTIONS:")
print(demo_df.head(20).to_string(index=False))

# PROJECT: A COMPUTER VISION/MACHINE LEARNING PIPELINE FOR DETECTING EYELID CANCER
# Supervisor: Prof. Khurshid Ahmad (Trinity College Dublin)
# ===================== IMPORTS =====================
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision.models import vit_b_16, ViT_B_16_Weights
import joblib
os.makedirs('saved_models', exist_ok=True)
# ===================== CONFIGURATION =====================
# Ensure results directory exists
RESULTS_DIR = 'training_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# GPU check
torch.backends.cudnn.benchmark = True
use_gpu = torch.cuda.is_available()
print(f"Using GPU: {use_gpu}")

# Initialize results storage
defn_columns = [
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
]
results = {col: [] for col in defn_columns}
training_histories = {}

# Matplotlib / seaborn styling
plt.style.use('default')
sns.set_palette("husl")

# ===================== VISUALIZATION/functions =====================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign','Malignant'], yticklabels=['Benign','Malignant'],
                ax=ax, cbar_kws={'label':'Count'})
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    out_fp = os.path.join(RESULTS_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(out_fp, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ===================== MODEL 1: CNN-A =====================
print("\n" + "="*80)
print("MODEL 1: CNN-A - CNN as Feature Extractor + Classifier")
print("="*80)

data_folder = "segmented_outputs_combined"
classes_map = {"benign":0, "malignant":1}
resize_dim = 224

# Prepare model
cnn_feat = models.resnet18(pretrained=True)
cnn_feat = nn.Sequential(*list(cnn_feat.children())[:-1])
cnn_feat.eval()
if use_gpu: cnn_feat = cnn_feat.cuda()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((resize_dim, resize_dim)),
    transforms.ToTensor()
])

# Collect paths & labels
split_paths, split_labels = {s:[] for s in ['train','val','test']}, {s:[] for s in ['train','val','test']}
for split in split_paths:
    for cls, lbl in classes_map.items():
        fp_list = glob(os.path.join(data_folder, split, cls, "*_foreground.png"))
        split_paths[split].extend(fp_list)
        split_labels[split].extend([lbl]*len(fp_list))

# Feature extraction
def extract_feats(paths):
    feats=[]
    for p in tqdm(paths, desc="Extracting CNN-A features"):
        img=cv2.imread(p)
        if img is None: continue
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t=transform(img).unsqueeze(0)
        if use_gpu: t=t.cuda()
        with torch.no_grad(): f=cnn_feat(t).cpu().numpy().flatten(); feats.append(f)
    return np.array(feats)

X_train = extract_feats(split_paths['train']); y_train=np.array(split_labels['train'])
X_val   = extract_feats(split_paths['val']);   y_val  =np.array(split_labels['val'])
X_test  = extract_feats(split_paths['test']);  y_test =np.array(split_labels['test'])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
plot_confusion_matrix(y_test, y_pred, "CNN-A")

print("\nCNN-A Results:", accuracy_score(y_test,y_pred))
rep=classification_report(y_test,y_pred,output_dict=True)
print(classification_report(y_test,y_pred))
# Save CNN-A RandomForest
joblib.dump(rf, 'saved_models/cnn_a_rf.pkl')
print("Saved CNN-A RandomForest to saved_models/cnn_a_rf.pkl")
# Record
results["Model"].append("CNN-A (Feature+RF)")
results["Accuracy"].append(accuracy_score(y_test, y_pred))
malignant_metrics = rep.get('1', {}) # Malignant class is 1
results["Precision"].append(malignant_metrics.get('precision', 0))
results["Recall"].append(malignant_metrics.get('recall', 0))
results["F1-Score"].append(malignant_metrics.get('f1-score', 0))

# ===================== MODEL 2: CNN-B =====================
print("\n" + "="*80)
print("MODEL 2: CNN-B - Direct Classifier on Segmented Lesion Foregrounds")
print("="*80)

batch_size, epochs = 8, 5

class LesionDataset(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels
        self.tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((resize_dim,resize_dim)), transforms.ToTensor()])
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        im=cv2.imread(self.paths[i]); im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        return self.tf(im), self.labels[i]

# prepare splits
train_p, val_p, test_p = [],[],[]
train_l, val_l, test_l = [],[],[]
for cls,lbl in classes_map.items():
    for split,lists in zip(['train','val','test'], [(train_p,train_l),(val_p,val_l),(test_p,test_l)]):
        fps=glob(os.path.join(data_folder,split,cls,"*_foreground.png"));
        lists[0].extend(fps); lists[1].extend([lbl]*len(fps))

dl_train = DataLoader(LesionDataset(train_p,train_l), batch_size=batch_size, shuffle=True)
dl_val   = DataLoader(LesionDataset(val_p,val_l),     batch_size=batch_size)
dl_test  = DataLoader(LesionDataset(test_p,test_l),   batch_size=batch_size)

# model init
cnn_b = models.resnet18(pretrained=True)
cnn_b.fc = nn.Linear(cnn_b.fc.in_features, 2)
if use_gpu: cnn_b = cnn_b.cuda()
crit = nn.CrossEntropyLoss(); opt = optim.Adam(cnn_b.parameters(), lr=1e-4)

hist = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
print(f"Train: {len(dl_train.dataset)}, Val: {len(dl_val.dataset)}, Test: {len(dl_test.dataset)}")
for ep in range(epochs):
    cnn_b.train(); tloss, correct, total = 0,0,0
    for imgs, lbls in tqdm(dl_train, desc=f"Epoch {ep+1}/{epochs}"):
        if use_gpu: imgs,lbls=imgs.cuda(),lbls.cuda()
        out=cnn_b(imgs); loss=crit(out,lbls); opt.zero_grad(); loss.backward(); opt.step()
        tloss+=loss.item(); preds=torch.argmax(out,dim=1); total+=lbls.size(0); correct+=(preds==lbls).sum().item()
    tr_acc=correct/total; hist['train_loss'].append(tloss/len(dl_train)); hist['train_acc'].append(tr_acc)
    # validation
    cnn_b.eval(); vloss, vcorr, vtot=0,0,0
    with torch.no_grad():
        for imgs, lbls in dl_val:
            if use_gpu: imgs,lbls=imgs.cuda(),lbls.cuda()
            out=cnn_b(imgs); loss=crit(out,lbls); vloss+=loss.item()
            preds=torch.argmax(out,dim=1); vtot+=lbls.size(0); vcorr+=(preds==lbls).sum().item()
    hist['val_loss'].append(vloss/len(dl_val)); hist['val_acc'].append(vcorr/vtot)
    print(f"Epoch {ep+1} -> Loss: {tloss/len(dl_train):.4f}/{vloss/len(dl_val):.4f}, Acc: {tr_acc:.4f}/{vcorr/vtot:.4f}")
training_histories['CNN-B'] = hist

# evaluate CNN-B
cnn_b.eval(); y_true, y_pred = [], []
with torch.no_grad():
    for imgs, lbls in tqdm(dl_test, desc="Evaluating CNN-B"):
        if use_gpu: imgs=imgs.cuda()
        out=cnn_b(imgs); preds=torch.argmax(out,dim=1).cpu().numpy(); y_pred.extend(preds); y_true.extend(lbls)
plot_confusion_matrix(y_true, y_pred, "CNN-B")
print("\nCNN-B Foreground Results:", accuracy_score(y_true,y_pred))
rep2=classification_report(y_true,y_pred,output_dict=True)
print(classification_report(y_true,y_pred))
# Save CNN-B model
torch.save(cnn_b.state_dict(), 'saved_models/cnn_b.pth')
print("Saved CNN-B ResNet model to saved_models/cnn_b.pth")
results["Model"].append("CNN-B (Foreground)")
results["Accuracy"].append(accuracy_score(y_true, y_pred))
malignant_metrics = rep2.get('1', {}) # Malignant class is 1
results["Precision"].append(malignant_metrics.get('precision', 0))
results["Recall"].append(malignant_metrics.get('recall', 0))
results["F1-Score"].append(malignant_metrics.get('f1-score', 0))

# ===================== FINAL COMPARISON =====================
print("\n" + "="*80)
print("FINAL MODEL COMPARISON")
print("="*80)
if results['Model']:
    df = pd.DataFrame(results)

    # --- Create Summary Table Image ---
    fig, ax = plt.subplots(figsize=(10, 4)) # Adjusted size for fewer columns
    ax.axis('off')
    table_data = [[mdl] + [f"{df.loc[i, col]:.3f}" for col in df.columns if col != 'Model'] for i, mdl in enumerate(df['Model'])]
    headers = df.columns
    tbl = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 1.5)
    for j in range(len(headers)):
        tbl[(0, j)].set_facecolor('#4CAF50')
        tbl[(0, j)].set_text_props(weight='bold', color='white')
    ax.set_title('Model Performance Summary', pad=20)
    fp_table = os.path.join(RESULTS_DIR, 'model_performance_table.png')
    plt.savefig(fp_table, dpi=300, bbox_inches='tight'); plt.close()

    # --- Create Comparison Plots (if more than one model) ---
    if len(df) > 1:
        # Accuracy Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(df['Model'], df['Accuracy'], alpha=0.7)
        ax.set_title('Accuracy Comparison'); ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
        for b, v in zip(bars, df['Accuracy']):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha='center')
        plt.tight_layout()
        fp_bar = os.path.join(RESULTS_DIR, 'accuracy_model_comparison.png')
        plt.savefig(fp_bar, dpi=300, bbox_inches='tight'); plt.close()

        # Radar Chart
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
        fp_radar = os.path.join(RESULTS_DIR, 'model_performance_radar.png')
        plt.savefig(fp_radar, dpi=300, bbox_inches='tight'); plt.close()

    # --- Print Final Summary to Console ---
    print("\nFINAL RESULTS SUMMARY:\n", df.to_string(index=False, float_format='%.3f'))
    best_idx = df['Accuracy'].idxmax()
    print(f"\nBest model: {df.loc[best_idx, 'Model']} (Accuracy: {df.loc[best_idx, 'Accuracy']:.3f})")
else:
    print("No models executed successfully.")

print("\nAnalysis complete. All outputs saved under the 'results' directory.")

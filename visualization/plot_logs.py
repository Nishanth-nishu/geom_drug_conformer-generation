import re
import matplotlib.pyplot as plt
import os

log_files = {
    '20k_mols': '/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/geom_baseline_2635278.log',
    '50k_mols': '/scratch/nishanth.r/nextmol_experiment/mol_expermiments/logs/geom_baseline_2635283.log'
}

out_dirs = [
    '/scratch/nishanth.r/nextmol_experiment/mol_expermiments/visualization',
    '/home2/nishanth.r/.gemini/antigravity-ide/brain/558b6e58-fdec-42b4-8aba-18a020748c84'
]

epoch_pattern = re.compile(r'Epoch\s+(\d+)/\d+\s+\|\s+train=([\d.]+)\s+\(mse=([\d.]+)\s+geo=([\d.]+)\)\s+val=([\d.]+)\s+\|\s+lr=([\d.e-]+)')
eval_epoch_pattern = re.compile(r'── GEOM-Drugs Eval \[Epoch (\d+)\]')
mat_r_pattern = re.compile(r'MAT-R\s+:\s+([\d.]+)\s+A')
cov_r_pattern = re.compile(r'COV-R@0.5A\s+:\s+([\d.]+)%')

data = {}

for label, log_file in log_files.items():
    epochs = []
    train_loss = []
    val_loss = []
    train_mse = []
    train_geo = []
    lr = []
    eval_metrics = {}

    with open(log_file, 'r') as f:
        lines = f.readlines()

    full_training_started = False
    current_eval_epoch = None

    for line in lines:
        if "Full training" in line:
            full_training_started = True
        
        if not full_training_started:
            continue

        m = epoch_pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            train_mse.append(float(m.group(3)))
            train_geo.append(float(m.group(4)))
            val_loss.append(float(m.group(5)))
            lr.append(float(m.group(6)))
        
        m_eval = eval_epoch_pattern.search(line)
        if m_eval:
            current_eval_epoch = int(m_eval.group(1))
            eval_metrics[current_eval_epoch] = {}
            continue
        
        if current_eval_epoch is not None:
            m_mat = mat_r_pattern.search(line)
            if m_mat and 'MAT-R' not in eval_metrics[current_eval_epoch]:
                eval_metrics[current_eval_epoch]['MAT-R'] = float(m_mat.group(1))
            
            m_cov = cov_r_pattern.search(line)
            if m_cov and 'COV-R' not in eval_metrics[current_eval_epoch]:
                eval_metrics[current_eval_epoch]['COV-R'] = float(m_cov.group(1))
            
            if "RMSD mean" in line:
                current_eval_epoch = None

    data[label] = {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_mse': train_mse,
        'train_geo': train_geo,
        'lr': lr,
        'eval_metrics': eval_metrics
    }

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# 1. Loss Curve
plt.figure(figsize=(10, 6))
for i, (label, d) in enumerate(data.items()):
    c = colors[i]
    plt.plot(d['epochs'], d['train_loss'], label=f'{label} Train Loss', alpha=0.8, color=c)
    plt.plot(d['epochs'], d['val_loss'], label=f'{label} Val Loss', alpha=0.8, color=c, linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
for d_out in out_dirs:
    if os.path.exists(d_out):
        plt.savefig(os.path.join(d_out, 'loss_curve_comparison.png'), dpi=150)
plt.close()

# 2. Eval Metrics Curve
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.set_xlabel('Epochs')
ax1.set_ylabel('MAT-R (Å)')
ax2.set_ylabel('COV-R@0.5Å (%)')

for i, (label, d) in enumerate(data.items()):
    eval_m = d['eval_metrics']
    if not eval_m: continue
    e_epochs = sorted(eval_m.keys())
    mat_r = [eval_m[e].get('MAT-R', 0) for e in e_epochs]
    cov_r = [eval_m[e].get('COV-R', 0) for e in e_epochs]
    
    c = colors[i]
    ax1.plot(e_epochs, mat_r, marker='o', label=f'{label} MAT-R', color=c, linestyle='-')
    ax2.plot(e_epochs, cov_r, marker='s', label=f'{label} COV-R', color=c, linestyle='--')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.title('Evaluation Metrics over Epochs')
ax1.grid(True, alpha=0.3)
fig.tight_layout()
for d_out in out_dirs:
    if os.path.exists(d_out):
        plt.savefig(os.path.join(d_out, 'eval_metrics_comparison.png'), dpi=150)
plt.close()

# 3. Learning Rate Curve
plt.figure(figsize=(10, 4))
for i, (label, d) in enumerate(data.items()):
    plt.plot(d['epochs'], d['lr'], label=label, color=colors[i])
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True)
plt.legend()
plt.tight_layout()
for d_out in out_dirs:
    if os.path.exists(d_out):
        plt.savefig(os.path.join(d_out, 'learning_rate_comparison.png'), dpi=150)
plt.close()

print("Comparison plots generated successfully.")

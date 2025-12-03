import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
# FairChem v2 
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core.datasets import AseReadMultiStructureDataset, data_list_collater

# ================= 
CHECKPOINT_NAME = "uma-s-1p1.pt"
TRAIN_FILENAME = "ymno3_train.xyz" 
VAL_FILENAME = "ymno3_val.xyz"
OUTPUT_MODEL_NAME = "uma_ymno3_finetuned.pt"
LOSS_PLOT_NAME = "training_loss.png"
TASK_NAME = "omat"
# ================= 

# Hyperparameters - REDUCED FOR CPU TESTING!!
LEARNING_RATE = 5e-5
EPOCHS = 2                
BATCH_SIZE = 2            
FORCE_COEFFICIENT = 100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_SUBSET_SIZE = 20    # Only 20 training samples for CPU test
VAL_SUBSET_SIZE = 5       # Only 5 validation samples for CPU test


def collate_fn(batch):
    """Collate function that adds dataset info for UMA"""
    batched = data_list_collater(batch)
    num_samples = len(batch)
    batched.dataset = [TASK_NAME] * num_samples
    return batched.to(DEVICE)


def load_uma_model(checkpoint_path):
    """Load UMA model using FairChem v2 API"""
    print(f"Loading UMA model: {checkpoint_path}...")
    predictor = load_predict_unit(checkpoint_path, device=str(DEVICE))
    model = predictor.model
    model.train()
    return model, predictor


def extract_tensor(value):
    """Extract tensor from value wth nested in dicts"""
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, dict):
        for key in ['output', 'pred', 'value', 'energy', 'forces']:
            if key in value:
                return extract_tensor(value[key])
        for v in value.values():
            if isinstance(v, torch.Tensor):
                return v
    return value


def plot_losses(train_losses, val_losses, save_path):
    """Plot and save train/val loss curves"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=8)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('UMA Fine-tuning: Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (train_l, val_l) in enumerate(zip(train_losses, val_losses)):
        plt.annotate(f'{train_l:.2f}', (epochs[i], train_l), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9, color='blue')
        plt.annotate(f'{val_l:.2f}', (epochs[i], val_l), textcoords="offset points", 
                     xytext=(0, -15), ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  -> Loss plot saved to: {save_path}")


def train():
    cwd = os.getcwd()
    
    print(f"Device: {DEVICE}")
    print("=" * 50)
    print(" QUICK TEST MODE - Using small data subset") # for CPU only
    print(f"   Train samples: {TRAIN_SUBSET_SIZE}, Val samples: {VAL_SUBSET_SIZE}")
    print("=" * 50)
    
    # 1. Load Data
    print("\nLoading datasets...")
    
    try:
        print(f"  -> Loading Train: {TRAIN_FILENAME}")
        full_train_dataset = AseReadMultiStructureDataset(
            config={
                "src": cwd,
                "pattern": TRAIN_FILENAME,
                "a2g_args": {"r_energy": True, "r_forces": True},
            }
        )
        
        print(f"  -> Loading Val: {VAL_FILENAME}")
        full_val_dataset = AseReadMultiStructureDataset(
            config={
                "src": cwd,
                "pattern": VAL_FILENAME,
                "a2g_args": {"r_energy": True, "r_forces": True},
            }
        )
        
        # CPU only: small subsets for testing
        train_indices = list(range(min(TRAIN_SUBSET_SIZE, len(full_train_dataset))))
        val_indices = list(range(min(VAL_SUBSET_SIZE, len(full_val_dataset))))
        
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_val_dataset, val_indices)
        
        print(f"  -> Using {len(train_dataset)} train, {len(val_dataset)} val samples")
        print(f"  -> Task: {TASK_NAME}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR loading datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"  -> {len(train_loader)} train batches, {len(val_loader)} val batches per epoch")

    # 2. Load Model
    checkpoint_path = os.path.join(cwd, CHECKPOINT_NAME)
    model, predictor = load_uma_model(checkpoint_path)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    loss_e_fn = nn.MSELoss()
    loss_f_fn = nn.MSELoss()

    best_val_loss = float('inf')
    
    energy_key = f"{TASK_NAME}_energy"
    forces_key = f"{TASK_NAME}_forces"

    # Track losses 
    train_losses = []
    val_losses = []
    print("\nStarting training...\n")
    
    # ====== training loop ======
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch)
            
            energy_pred = extract_tensor(out[energy_key])
            forces_pred = extract_tensor(out[forces_key])
            
            loss_e = loss_e_fn(energy_pred, batch.energy)
            loss_f = loss_f_fn(forces_pred, batch.forces)
            loss = loss_e + (FORCE_COEFFICIENT * loss_f)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ====== validation loop ======
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                out = model(batch)
                
                energy_pred = extract_tensor(out[energy_key])
                forces_pred = extract_tensor(out[forces_key])
                    
                loss_e = loss_e_fn(energy_pred, batch.energy)
                loss_f = loss_f_fn(forces_pred, batch.forces)
                val_loss += (loss_e + (FORCE_COEFFICIENT * loss_f)).item()
                val_batches += 1
        
        avg_train = train_loss / max(num_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        
        # save losses
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), OUTPUT_MODEL_NAME)
            print("New model saved.")
    
    # Plot
    print("\nGenerating loss plot...")
    plot_losses(train_losses, val_losses, LOSS_PLOT_NAME)
    
    print("\n" + "=" * 50)
    print("✓ Test training complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {OUTPUT_MODEL_NAME}")
    print(f"  Loss plot saved to: {LOSS_PLOT_NAME}")

if __name__ == "__main__":
    train()
"""
ANFIS Training Module
=====================
Training metode refaktorisane iz originalnog koda:
- _convert_to_pytorch
- forward_torch
- train_hybrid (bazirano na train_lse_only + premise gradovi)
- evaluate
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow.pyfunc
from sklearn.metrics import roc_auc_score
from src.anfis.core import ANFISAdvanced


def convert_to_pytorch(anfis_model):
    """
    Konvertuje NumPy parametre u PyTorch tensore sa gradient tracking.
    Refaktor _convert_to_pytorch.
    """
    print("\n🔧 Converting to PyTorch...")

    # Premise parameters (Layer 1) - MF centers i widths
    anfis_model.mf_params_torch = []

    for i in range(anfis_model.n_inputs):
        params_tensor = torch.tensor(
            anfis_model.mf_params[i],
            dtype=torch.float32,
            requires_grad=True
        )
        anfis_model.mf_params_torch.append(params_tensor)

    # Consequent parameters (Layer 4)
    anfis_model.consequent_params_torch = torch.tensor(
        anfis_model.consequent_params,
        dtype=torch.float32,
        requires_grad=True
    )

    print(f" ✅ Premise parameters: {len(anfis_model.mf_params_torch)} inputs")
    print(f" ✅ Consequent parameters: {anfis_model.consequent_params_torch.shape}")


def forward_torch(anfis_model, X_torch):
    """
    PyTorch forward pass - automatski prati gradijente!
    Refaktor forward_torch iz originalnog koda.

    Args:
        anfis_model: instance ANFISAdvanced
        X_torch: Input tensor shape (n_samples, n_inputs)

    Returns:
        output: Final output tensor shape (n_samples,)
    """
    n_samples = X_torch.shape[0]

    # LAYER 1: FUZZIFICATION (PyTorch verzija, Gaussian-only kao u originalu)
    mu = []
    for i in range(anfis_model.n_inputs):
        n_mfs = anfis_model.n_mfs_per_input[i]
        mu_input = torch.zeros(n_samples, n_mfs)
        for j in range(n_mfs):
            center = anfis_model.mf_params_torch[i][j, 0]
            sigma = anfis_model.mf_params_torch[i][j, 1]
            mu_input[:, j] = torch.exp(-0.5 * ((X_torch[:, i] - center) / sigma) ** 2)
        mu.append(mu_input)

    # LAYER 2: RULE FIRING
    w = torch.ones(n_samples, anfis_model.n_rules)
    for rule_idx, rule in enumerate(anfis_model.rule_base):
        for input_idx, mf_idx in enumerate(rule):
            w[:, rule_idx] *= mu[input_idx][:, mf_idx]

    # LAYER 3: NORMALIZATION
    w_sum = w.sum(dim=1, keepdim=True)
    w_sum = torch.where(w_sum == 0, torch.ones_like(w_sum), w_sum)
    w_bar = w / w_sum

    # LAYER 4: CONSEQUENT
    # f = X · θ + b
    X_expanded = X_torch.unsqueeze(1).expand(-1, anfis_model.n_rules, -1)
    weights = anfis_model.consequent_params_torch[:, :-1].unsqueeze(0)
    bias = anfis_model.consequent_params_torch[:, -1].unsqueeze(0)
    f = (X_expanded * weights).sum(dim=2) + bias  # (n_samples, n_rules)

    # LAYER 5: WEIGHTED SUM
    output = (w_bar * f).sum(dim=1)

    return output


def train_hybrid(
    anfis_model,
    X_train,
    y_train,
    premise_training = True,
    epochs=100,
    lr_premise=1e-3,
    lr_consequent=1e-2,
    batch_size=32,
    verbose=True
):
    """
    FULL HYBRID TRAINING:
    - Premise parametri (mf_params_torch) se treniraju sa manjim LR.
    - Consequent parametri (consequent_params_torch) sa većim LR.
    Bazirano na train_lse_only, ali sada su i premise trainable.

    Args:
        anfis_model: ANFISAdvanced instance
        X_train, y_train: NumPy arrays
        epochs: broj epoha
        lr_premise: learning rate za MF parametre
        lr_consequent: learning rate za consequent parametre
        batch_size: batch size
        verbose: ispis napretka

    Returns:
        history: dict sa 'loss' i 'epoch'
    """
    print("\n" + "="*70)
    print("🔥 ANFIS HYBRID TRAINING (Premise + Consequent)")
    print("="*70)

    # Postavi premise mode
    # ✅ 
    if premise_training:
        anfis_model.enable_premise_training()
    else:
        print("🔒 Fixed premise parameters")
    
    # Optimizer (samo trainable params)
    trainable_params = anfis_model.get_trainable_params()
    optimizer = optim.Adam(trainable_params)

    # Konverzija u PyTorch
    if not hasattr(anfis_model, 'consequent_params_torch'):
        convert_to_pytorch(anfis_model)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Parametri za trening
    params = [
        {'params': anfis_model.consequent_params_torch, 'lr': lr_consequent},
    ]
    # Premise parametri kao posebna grupa
    for p in anfis_model.mf_params_torch:
        params.append({'params': p, 'lr': lr_premise})

    optimizer = optim.Adam(params)
    criterion = nn.MSELoss()

    history = {'loss': [], 'epoch': []}
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"\n📊 Training Configuration:")
    print(f"   Samples: {n_samples}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   LR premise: {lr_premise}")
    print(f"   LR consequent: {lr_consequent}")
    print(f"   Premise param groups: {len(anfis_model.mf_params_torch)}")

    print(f"\n{'Epoch':<10} {'Loss':<15} {'Progress'}")
    print("-"*70)

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            X_batch = X_tensor[start_idx:end_idx]
            y_batch = y_tensor[start_idx:end_idx]

            # Forward
            y_pred = forward_torch(anfis_model, X_batch)

            # Loss
            loss = criterion(y_pred, y_batch)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * (end_idx - start_idx)

        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)
        history['epoch'].append(epoch + 1)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            progress_bar = '█' * int((epoch + 1) / epochs * 30)
            print(f"{epoch+1:<10} {epoch_loss:<15.6f} {progress_bar}")


    # Sync nazad u NumPy
    anfis_model.consequent_params = anfis_model.consequent_params_torch.detach().numpy()
    anfis_model.mf_params = [p.detach().numpy() for p in anfis_model.mf_params_torch]

    anfis_model.sync_params_from_torch()  # ← Sinhronizuj nazad
    
    print(f"✅ Training complete | Premise training: {premise_training}")
    return history


def evaluate(anfis_model, X_test, y_test):
    """
    Evaluiraj model na test podacima.
    Refaktor evaluate iz originalnog koda.

    Args:
        anfis_model: ANFISAdvanced instance
        X_test, y_test: NumPy arrays

    Returns:
        metrics: dict sa MSE, RMSE, MAE, R2
    """
    # Inference (koristi PyTorch ili čist NumPy forward)
    from numpy import ndarray
    if isinstance(X_test, np.ndarray):
        # ako imamo PyTorch parametre, koristi forward_torch
        if hasattr(anfis_model, 'consequent_params_torch'):
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            with torch.no_grad():
                y_pred_t = forward_torch(anfis_model, X_tensor)
            y_pred = y_pred_t.cpu().numpy()
        else:
            y_pred, _ = anfis_model.forward(X_test)
    else:
        # ako već dobiješ tensor
        with torch.no_grad():
            y_pred_t = forward_torch(anfis_model, X_test)
        y_pred = y_pred_t.cpu().numpy()

    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    print("\n" + "="*70)
    print("📊 EVALUATION METRICS")
    print("="*70)
    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   R²:   {r2:.6f}")
    print("="*70)

    return metrics


class ANFISPyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, anfis_model):
        self.anfis_model = anfis_model
    
    def predict(self, context, model_input):
        if hasattr(self.anfis_model, 'consequent_params_torch'):
            X_torch = torch.tensor(model_input.values, dtype=torch.float32)
            with torch.no_grad():
                y_pred_t = forward_torch(self.anfis_model, X_torch)
            return y_pred_t.cpu().numpy().reshape(-1, 1)
        else:
            return self.anfis_model.forward(model_input.values)[0]

def mlflow_train_anfis(X_train, y_train, X_test, y_test, params):  # ← DODAJ X_test, y_test
    """MLflow-compatible ANFIS training"""
    with mlflow.start_run(run_name="anfis_hybrid"):
        mlflow.log_params(params)
        
        model = ANFISAdvanced(n_inputs=X_train.shape[1], **params)
        convert_to_pytorch(model)
        history = train_hybrid(model, X_train, y_train, **params)
        
        # Evaluacija
        metrics = evaluate(model, X_test, y_test)  # ← Sada defined
        mlflow.log_metric("final_loss", history['loss'][-1])
        mlflow.log_metric("r2_score", metrics['r2'])
        
        # Log model
        pyfunc_model = ANFISPyFunc(model)
        mlflow.pyfunc.log_model(
            "anfis_model", 
            python_model=pyfunc_model,
            input_example=X_test[:5]
        )
        
        return model, metrics
    
def set_premise_training(self, enable: bool = True):
    """
    Kontroliraj premise training:
    enable=True: Premise trainable (adaptivni MF)
    enable=False: Premise fixed (domain knowledge only)
    """
    self.premise_training_enabled = enable
    
    if enable:
        print("🔧 Premise training ENABLED (adaptivni MF centri/širine)")
        self.enable_premise_training()  # Tvoja postojeća funkcija
    else:
        print("🔒 Premise training DISABLED (fixed ISO/OSHA init)")
        if hasattr(self, 'mf_params_torch'):
            # Zaustavi gradijente
            for p in self.mf_params_torch:
                p.requires_grad = False
    
    return self.premise_training_enabled
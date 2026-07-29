import torch
from utils import load_model

def check_model_weights(model_path: str):
    print(f"Loading {model_path}...")
    model = load_model(model_path)
    
    print("\n--- Inspecting PyTorch Meta Head ---")
    
    nan_found = False
    
    # Check MLP layers (Linear weights and biases)
    if hasattr(model, 'mlp') and model.mlp is not None:
        for name, param in model.mlp.named_parameters():
            nan_count = torch.isnan(param).sum().item()
            total_params = param.numel()
            if nan_count > 0:
                print(f"[!] MLP {name}: {nan_count}/{total_params} values are NaN!")
                nan_found = True
            else:
                print(f"[✓] MLP {name}: Clean (0 NaNs).")
    else:
        print("No MLP found in model.")

    # Check Embedding layer
    if hasattr(model, 'head') and model.head is not None:
        for name, param in model.head.named_parameters():
            nan_count = torch.isnan(param).sum().item()
            total_params = param.numel()
            if nan_count > 0:
                print(f"[!] Embedding {name}: {nan_count}/{total_params} values are NaN!")
                nan_found = True
            else:
                print(f"[✓] Embedding {name}: Clean (0 NaNs).")
    else:
        print("No Embedding head found in model.")

    print("\n--- Diagnostic Conclusion ---")
    if nan_found:
        print("Verdict: The PyTorch meta head weights HAVE EXPLODED into NaNs.")
        print("Cause: Unfiltered NaN targets during F.mse_loss computation.")
        print("Action: Implement the NaN filtering fix in _train_meta_head and retrain the meta head.")
    else:
        print("Verdict: The weights are clean.")
        print("Cause: The issue is happening dynamically during inference, not from corrupted weights.")

if __name__ == "__main__":
    # Point this to the model you just evaluated
    model_file = "models/dynamic_ensemble_20260728_034545.pkl"
    check_model_weights(model_file)
import os
import torch
from src.verification.multi_hop_reasoner import MultiHopReasoner

def test_persistence():
    print("Testing GNN persistence...")
    
    # 1. Initialize a model
    model = MultiHopReasoner(hidden_dim=32, num_layers=2)
    model.eval()
    
    # 2. Save it
    path = "models/test_persistence_model.pt"
    model.save_model(path)
    
    # 3. Load it
    loaded_model = MultiHopReasoner.load_model(path)
    
    # 4. Compare parameters
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        if not torch.equal(p1, p2):
            print("Parameter mismatch!")
            return False
            
    print("Model weights verified!")
    
    # 5. Check config
    if loaded_model.gnn.input_proj.out_features != 32:
        print(f"Config mismatch! Expected 32, got {loaded_model.gnn.input_proj.out_features}")
        return False
        
    print("Config verified!")
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)
        
    return True

if __name__ == "__main__":
    if test_persistence():
        print("\nALL PERSISTENCE TESTS PASSED")
    else:
        print("\nTESTS FAILED")
        exit(1)

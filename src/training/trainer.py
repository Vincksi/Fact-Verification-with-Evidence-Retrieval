"""
Trainer for GNN-based verification model.
"""

import threading
import time
from typing import Dict

import torch
import torch.optim as optim

from src.verification.multi_hop_reasoner import MultiHopReasoner


class GATTrainer:
    """
    Manages the training process for the GAT model.
    Runs in a background thread to avoid blocking the API.
    """

    def __init__(self, pipeline, learning_rate: float = 1e-4):
        self.pipeline = pipeline
        self.learning_rate = learning_rate
        self.is_training = False
        self.stop_requested = False
        self.thread = None

        # Training metrics
        self.metrics = {
            "epoch": 0,
            "total_epochs": 0,
            "current_loss": 0.0,
            "avg_loss": 0.0,
            "accuracy": 0.0,
            "history": []  # List of {epoch, loss, accuracy}
        }

        self.label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2}

    def start_training(self, epochs: int = 5):
        """Start training in a background thread."""
        if self.is_training:
            return False

        self.is_training = True
        self.stop_requested = False
        self.metrics["total_epochs"] = epochs
        self.metrics["epoch"] = 0
        self.metrics["history"] = []

        self.thread = threading.Thread(target=self._run_training, args=(epochs,))
        self.thread.daemon = True
        self.thread.start()
        return True

    def stop_training(self):
        """Request training to stop."""
        self.stop_requested = True

    def get_status(self) -> Dict:
        """Get current training status and metrics."""
        return {
            "is_training": self.is_training,
            "metrics": self.metrics
        }

    def _run_training(self, epochs: int):
        """Internal training loop."""
        try:
            model = self.pipeline.verifier
            if not isinstance(model, MultiHopReasoner):
                print("Error: Verifier is not a MultiHopReasoner")
                self.is_training = False
                return

            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            # Load training claims
            claims = self.pipeline.dataset.load_claims("train")
            if not claims:
                print("Error: No training claims found")
                self.is_training = False
                return

            # For demo purposes/speed, we might want to limit the samples if it's too slow
            # but let's try with full data first or a reasonable subset.
            # claims = claims[:100]

            for epoch in range(1, epochs + 1):
                if self.stop_requested:
                    break

                self.metrics["epoch"] = epoch
                epoch_loss = 0.0
                correct = 0
                processed = 0

                # Shuffle claims
                import random
                random.shuffle(claims)

                from tqdm import tqdm
                pbar = tqdm(claims, desc=f"Epoch {epoch}/{epochs}", leave=False)
                for claim in pbar:
                    if self.stop_requested:
                        break

                    # Get evidence sentences
                    evidence_dict = self.pipeline.dataset.get_evidence_sentences(claim)
                    all_sentences = []
                    for doc_sentences in evidence_dict.values():
                        all_sentences.extend(doc_sentences)

                    if not all_sentences:
                        continue

                    # Prepare label
                    label_idx = self.label_map.get(claim.label, 2)

                    # Periodic update to current_loss for UI
                    loss = model.train_step(claim.claim, all_sentences, label_idx, optimizer)
                    epoch_loss += loss

                    # Calculate accuracy (evaluation mode)
                    model.eval()
                    with torch.no_grad():
                        pred_label, _, _, _ = model.predict(claim.claim, all_sentences)
                        if pred_label == claim.label:
                            correct += 1

                    processed += 1
                    self.metrics["current_loss"] = loss

                    # Update progress bar
                    if processed % 10 == 0:
                        pbar.set_postfix({
                            "loss": f"{loss:.4f}",
                            "acc": f"{correct / processed:.2%}"
                        })

                    # Yield slightly to allow other things to happen
                    time.sleep(0.001)

                if processed > 0:
                    avg_loss = epoch_loss / processed
                    accuracy = correct / processed
                    self.metrics["avg_loss"] = avg_loss
                    self.metrics["accuracy"] = accuracy
                    self.metrics["history"].append({
                        "epoch": epoch,
                        "loss": avg_loss,
                        "accuracy": accuracy
                    })

            print(f"Training finished: Epoch {self.metrics['epoch']}")
            
            # Save the final model
            model_path = "models/gat_model.pt"
            model.save_model(model_path)
            print(f"Trained model persisted to {model_path}")

        except Exception as e:
            print(f"Training error: {e}")
        finally:
            self.is_training = False

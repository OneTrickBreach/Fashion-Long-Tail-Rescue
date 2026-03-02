"""
evaluate_cold_start.py
======================
Simulates cold start "new arrival" recommendations.
"""
import os
import json
import torch
import random
import pandas as pd
import numpy as np

from src.utils.helpers import load_config, get_device
from src.data.dataset import build_id_maps, load_multimodal_embeddings
from src.villain.model import VillainModel
from src.hero.model import HeroModel

def get_eval_items_with_target(user_seqs, mode="test"):
    # Reproduce validation/test inputs
    samples = []
    for uid, raw_items in user_seqs.items():
        if len(raw_items) < 3: continue
        if mode == "test":
            samples.append((uid, raw_items[:-1], raw_items[-1]))
        elif mode == "val":
            samples.append((uid, raw_items[:-2], raw_items[-2]))
    return samples

def run_cold_start_simulation(config):
    device = get_device(config["embedding"]["device"])
    paths = config["paths"]
    
    art_path = os.path.join(paths["sampled_data"], "articles_sampled.csv")
    txn_path = os.path.join(paths["sampled_data"], "transactions_sampled.csv")
    
    print("Building sequences...")
    txn = pd.read_csv(txn_path, parse_dates=["t_dat"]).sort_values(["customer_id", "t_dat"])
    user_sequences = {uid: grp["article_id"].tolist() for uid, grp in txn.groupby("customer_id")}
    
    id_to_idx, idx_to_id = build_id_maps(art_path)
    num_items = max(id_to_idx.values()) + 1
    
    # Identify items in training vs test
    train_items = set()
    for seq in user_sequences.values():
        if len(seq) >= 3:
            for item in seq[:-2]:
                if item in id_to_idx:
                    train_items.add(id_to_idx[item])
                    
    test_samples = get_eval_items_with_target(user_sequences, "test")
    test_valid_samples = []
    
    for uid, inp, tgt in test_samples:
        if tgt in id_to_idx:
            test_valid_samples.append((uid, inp, id_to_idx[tgt]))
            
    # Filter test samples down to ONLY those where target was never in train
    cold_start_samples = [s for s in test_valid_samples if s[2] not in train_items]
    
    print(f"Total test interactions involving purely cold-start items: {len(cold_start_samples)}")
    
    random.seed(42)
    # pick up to 100 for simulation
    eval_samples = random.sample(cold_start_samples, min(100, len(cold_start_samples)))
    
    # Load Multimodal embeddings
    emb_path = os.path.join(paths["embeddings"], "multimodal_embeddings.pt")
    visual_embeddings = load_multimodal_embeddings(emb_path, id_to_idx, num_items).to(device)
    
    # Load Models
    print("Loading Villain model...")
    villain = VillainModel(num_items=num_items, config=config["villain"]).to(device)
    villain.load_state_dict(torch.load(os.path.join(paths["checkpoints"], "villain_latest.pt"), map_location=device, weights_only=False)["model_state_dict"])
    villain.eval()
    
    print("Loading Hero model...")
    hero = HeroModel(num_items=num_items, config=config).to(device)
    hero.load_state_dict(torch.load(os.path.join(paths["checkpoints"], "hero_best.pt"), map_location=device, weights_only=False)["model_state_dict"])
    hero.eval()
    
    print("\nSimulating Cold Start Performance...")
    
    villain_ranks = []
    hero_ranks = []
    qualitative_examples = []
    
    with torch.no_grad():
        # Inject visuals into the catalog prediction matrix!
        hero_catalog_vis = hero.visual_proj(visual_embeddings) 
        hero_full_catalog = hero.item_emb.weight + hero_catalog_vis
        
        for idx, (uid, seq, target_idx) in enumerate(eval_samples):
            # Prep inputs
            # Map raw IDs to indices, truncate to max_seq_len, pad
            idx_seq = [id_to_idx[a] for a in seq if a in id_to_idx][-config["hero"]["max_seq_len"]:]
            slen = len(idx_seq)
            padded = idx_seq + [0] * (config["hero"]["max_seq_len"] - slen)
            
            item_seq = torch.tensor([padded], dtype=torch.long).to(device)
            positions = torch.arange(config["hero"]["max_seq_len"], dtype=torch.long).unsqueeze(0).to(device)
            seq_len_t = torch.tensor([slen], dtype=torch.long).to(device)
            vis_seq = visual_embeddings[item_seq]
            
            # Villain Prediction
            v_logits = villain(item_seq, positions, seq_len_t) # (1, num_items)
            v_logits[0, 0] = float("-inf")
            # Get rank of target
            # Sort descending
            v_sorted = torch.argsort(v_logits[0], descending=True)
            v_rank = (v_sorted == target_idx).nonzero(as_tuple=True)[0].item() + 1
            villain_ranks.append(v_rank)
            
            # Hero Prediction
            _, h_states = hero(item_seq, positions, vis_seq) # h_states: (1, H)
            h_logits = torch.matmul(h_states, hero_full_catalog.transpose(0, 1)) # (1, num_items)
            h_logits[0, 0] = float("-inf")
            
            h_sorted = torch.argsort(h_logits[0], descending=True)
            h_rank = (h_sorted == target_idx).nonzero(as_tuple=True)[0].item() + 1
            hero_ranks.append(h_rank)
            
            # Save qualitative highlight if Hero massively outperformed Villain
            if h_rank <= 12 and v_rank > 500:
                qualitative_examples.append({
                    "customer_id": uid,
                    "target_article_id": idx_to_id[target_idx],
                    "hero_rank": h_rank,
                    "villain_rank": v_rank
                })
                
    # Metrics
    v_ranks = np.array(villain_ranks)
    h_ranks = np.array(hero_ranks)
    
    results = {
        "num_eval_samples": len(eval_samples),
        "villain": {
            "avg_rank": float(np.mean(v_ranks)),
            "hit_at_12": float(np.mean(v_ranks <= 12)),
            "hit_at_50": float(np.mean(v_ranks <= 50))
        },
        "hero": {
            "avg_rank": float(np.mean(h_ranks)),
            "hit_at_12": float(np.mean(h_ranks <= 12)),
            "hit_at_50": float(np.mean(h_ranks <= 50))
        },
        "qualitative_examples": qualitative_examples[:5]
    }
    
    print("\nCold Start Results:")
    print(f"Villain Avg Rank: {results['villain']['avg_rank']:.1f} (Hit@12: {results['villain']['hit_at_12']:.1%})")
    print(f"Hero Avg Rank:    {results['hero']['avg_rank']:.1f} (Hit@12: {results['hero']['hit_at_12']:.1%})")
    print(f"Found {len(qualitative_examples)} qualitative highlight examples.")
    
    out_path = os.path.join(paths["outputs"], "hero_cold_start_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    cfg = load_config()
    run_cold_start_simulation(cfg)

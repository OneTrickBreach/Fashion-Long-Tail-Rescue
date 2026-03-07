"""
aha_testcase.py
##############################################################
Find the most compelling "Aha!" example:
  - A niche item the Villain buried (rank 500+)
  - That the Hero surfaced in the Top 12
Actual test case included in comments at end of file.
##############################################################

Run from src:
    python -m src.aha_testcase
"""

import torch
import pandas as pd

from src.utils.helpers import load_config, set_seed, get_device
from src.data.dataset import build_dataloaders
from src.villain.config import get_villain_config
from src.villain.model import VillainModel
from src.hero.config import get_hero_config
from src.hero.model import HeroModel

#Load config and set env variables
cfg = load_config()
set_seed(cfg["project"]["seed"])
device = get_device(cfg["embedding"]["device"])
eval_k = cfg["evaluation"]["k"]  # 12

#Load test set
print("Building dataloaders...")
_, _, test_loader, meta = build_dataloaders(cfg, mode="villain")
num_items = meta["num_items"]
idx_to_id = meta["idx_to_id"]

#article.csv metadata and purchase counts
articles = pd.read_csv("data/sampled/articles_sampled.csv").set_index("article_id")
txn = pd.read_csv("data/sampled/transactions_sampled.csv")
article_counts = txn["article_id"].value_counts()

#Load Villain
print("Loading Villain model...")
villain_cfg = get_villain_config(cfg)
villain = VillainModel(num_items=num_items, config=villain_cfg).to(device)
ckpt_v = torch.load("checkpoints/villain_best.pt", map_location=device, weights_only=False)
villain.load_state_dict(ckpt_v["model_state_dict"])
villain.eval()

#Load Hero
print("Loading Hero model...")
hero_cfg = get_hero_config(cfg)
hero = HeroModel(config=cfg, num_items=num_items).to(device)
ckpt_h = torch.load("checkpoints/hero_best.pt", map_location=device, weights_only=False)
hero.load_state_dict(ckpt_h["model_state_dict"])
hero.eval()

#Load visual embeddings
print("Loading visual embeddings...")
visual_embeddings = torch.load(
    "data/embeddings/multimodal_embeddings.pt", map_location=device
)  # (num_items, 2048)

#Search for items within criteria
best = None

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        print(batch.keys())
        break

    for batch_idx, batch in enumerate(test_loader):
        item_seq  = batch["item_seq"].to(device)
        positions = batch["positions"].to(device)
        targets   = batch["target"].to(device)
        seq_len   = batch["seq_len"].to(device)
        vis_seq   = batch.get("visual_embeds")
        if vis_seq is not None:
            vis_seq = vis_seq.to(device)

        #Villain logits
        v_logits = villain(item_seq, positions, seq_len)
        v_logits[:, 0] = float("-inf")

        #Hero logits
        h_logits, _ = hero(item_seq, positions, vis_seq)
        h_logits[:, 0] = float("-inf")

        #Created full ranked list
        v_ranked = v_logits.argsort(dim=-1, descending=True)  # (B, num_items)
        h_ranked = h_logits.argsort(dim=-1, descending=True)  # (B, num_items)

        customer_ids = batch.get("customer_id", [None] * len(targets))

        for i, target in enumerate(targets.cpu().tolist()):
            # Find rank of target in each model (1-indexed)
            v_rank = (v_ranked[i] == target).nonzero(as_tuple=True)[0].item() + 1
            h_rank = (h_ranked[i] == target).nonzero(as_tuple=True)[0].item() + 1

            # Criteria: Hero in top 12, Villain buried at 500+
            if h_rank <= eval_k and v_rank >= 500:
                article_id = idx_to_id.get(target, target)
                purchase_count = article_counts.get(article_id, 0)
                score = v_rank - h_rank  # higher = more dramatic

                #Replace best
                if best is None or score > best["score"]:
                    cid = customer_ids[i]
                    best = {
                        "customer_id": cid.item() if hasattr(cid, "item") else cid,
                        "article_idx": target,
                        "article_id": article_id,
                        "villain_rank": v_rank,
                        "hero_rank": h_rank,
                        "purchase_count": int(purchase_count),
                        "score": score,
                    }

        if batch_idx % 20 == 0:
            found = f"villain #{best['villain_rank']} → hero #{best['hero_rank']}" if best else "none yet"
            print(f"  Batch {batch_idx}/{len(test_loader)} — best: {found}")

#Print result
print("\n" + "=" * 55)
if best is None:
    print("No example found. Try lowering Villain threshold to 200+.")
else:
    print("THE TESTCASE")
    print("=" * 55)
    print(f"  Customer ID:      {best['customer_id']}")
    print(f"  Article ID:       {best['article_id']}")
    print(f"  Purchase count:   {best['purchase_count']} total purchases (long-tail)")
    print(f"  Villain rank:     #{best['villain_rank']:,}  ← completely buried")
    print(f"  Hero rank:        #{best['hero_rank']}  ← Top 12 ✓")

    article_id = best["article_id"]
    if article_id in articles.index:
        row = articles.loc[article_id]
        print(f"\n  Item details:")
        for col in ["prod_name", "product_type_name", "colour_group_name", "department_name"]:
            if col in articles.columns:
                print(f"    {col}: {row[col]}")
print("=" * 55)

#Print user's purchase history
print("\n  Purchase history:")
user_txn = txn[txn["customer_id"] == best["customer_id"]].sort_values("t_dat")
for _, row in user_txn.iterrows():
    aid = row["article_id"]
    if aid in articles.index:
        item = articles.loc[aid]
        print(f"    {row['t_dat'].date() if hasattr(row['t_dat'], 'date') else row['t_dat']} — "
              f"{item.get('prod_name', aid)} "
              f"({item.get('colour_group_name', '')})")
    else:
        print(f"    {row['t_dat']} — article {aid}")

'''
=======================================================
THE TESTCASE
=======================================================
  Customer ID:      a20b9b0f62d3059c35c779f1c97ff5f7120b6a416e13be31dcd0a719a5a3b611
  Article ID:       820572011
  Purchase count:   19 total purchases (long-tail)
  Villain rank:     #10,837  ← completely buried
  Hero rank:        #3  ← Top 12 ✓

  Item details:
    prod_name: Liza Padded (Milano) 2pk
    product_type_name: Bra
    colour_group_name: White
    department_name: Casual Lingerie
=======================================================

  Purchase history:
    2020-08-03 — Jelly tunic (Yellow)
    2020-08-03 — Rio romantic tunic (Green)
    2020-08-03 — Moma (White)
    2020-08-03 — Kaia (Yellow)
    2020-08-03 — Liza Padded (Milano) 2pk (White)
    2020-08-03 — Curly. (Dark Blue)
    2020-08-12 — Miso (Dark Beige)
    2020-08-12 — Low Price l/l pj BB (Dark Blue)
    2020-08-12 — Liza Padded (Milano) 2pk (White)
'''

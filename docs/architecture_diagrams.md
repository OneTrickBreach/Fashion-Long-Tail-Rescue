# Architecture Diagrams (Mermaid)

> **How to export as PNG:**
> 1. Go to [mermaid.live](https://mermaid.live)
> 2. Paste a diagram block below into the editor
> 3. Click **Actions → Export PNG** (or SVG)
> 4. Save to `analytics/presentation/`

---

## ACT 1 — VILLAIN: SASRec + Position Bias

```mermaid
flowchart TD
    subgraph Input["📥 Input"]
        A["item_seq\n(B, S)"]
        B["positions\n(B, S)"]
    end

    subgraph Embeddings["🔢 Embedding Layer"]
        C["nn.Embedding\n(V, D)"]
        D["nn.Embedding\n(S, D)"]
    end

    A --> C --> E["item_emb\n(B, S, D)"]
    B --> D --> F["pos_emb\n(B, S, D)"]

    E --> G["➕ Add + LayerNorm"]
    F --> G
    G --> H["x\n(B, S, D)"]
    H --> I["Dropout(0.2)"]

    subgraph Transformer["🔄 TransformerEncoder × 3 layers"]
        J["MultiHeadAttention\n(D, H=4)"]
        K["causal_mask (S, S)\npad_mask (B, S)"]
        J --- K
    end

    I --> Transformer
    Transformer --> L["x_out\n(B, S, D)"]
    L --> M["gather\n(seq_len - 1)"]
    M --> N["last_hidden\n(B, D)"]
    N --> O["LayerNorm"]
    O --> P["last_hidden @\nitem_embed.weight.T"]
    P --> Q["logits\n(B, V)"]
    Q --> R["× pop_bias\n(V,)"]
    R --> S["🎯 final_logits\n(B, V) = (256, 26933)"]

    style Input fill:#E63946,color:#fff,stroke:#333
    style Embeddings fill:#457B9D,color:#fff,stroke:#333
    style Transformer fill:#2A9D8F,color:#fff,stroke:#333
    style S fill:#E9C46A,color:#333,stroke:#333
```

---

## ACT 2 — HERO: BST + ResNet50 Visual Fusion + Contrastive



```mermaid
flowchart TD
    subgraph Input["📥 Input"]
        A["item_seq\n(B, S)"]
        B["positions\n(B, S)"]
        V["visual_embeds\n(B, S, 2048)"]
    end

    subgraph IDEmbed["🔢 ID Embeddings"]
        C["nn.Embedding(V, D)"]
        D["nn.Embedding(S, D)"]
    end

    A --> C --> E["item_emb (B, S, D)"]
    B --> D --> F["pos_emb (B, S, D)"]
    E --> G["➕ Add"]
    F --> G
    G --> H["seq_repr (B, S, D)"]

    subgraph VisualProj["👁️ Visual Projection"]
        VP1["Linear(2048 → D)"]
        VP2["LayerNorm(D)"]
        VP3["Dropout(0.1)"]
        VP1 --> VP2 --> VP3
    end

    V --> VisualProj
    VisualProj --> VOut["v_proj (B, S, D)"]

    H --> Fuse["➕ seq_repr + v_proj"]
    VOut --> Fuse
    Fuse --> FN["fusion_norm\n(LayerNorm)"]
    FN --> Fused["fused (B, S, D)"]
    Fused --> Drop["Dropout(0.1)"]

    subgraph Transformer["🔄 TransformerEncoder × 3 layers"]
        T1["MultiHeadAttention (D, H=4)"]
        T2["FFN: D → 4D → D (GELU)"]
        T3["causal_mask (S, S)\npad_mask (B, S)"]
        T1 --- T2 --- T3
    end

    Drop --> Transformer
    Transformer --> Enc["encoded_seq (B, S, D)"]
    Enc --> Gather["gather (last non-PAD)"]
    Gather --> FS["final_states (B, D)"]

    FS --> Logits["final_states @\nitem_emb.weight.T"]
    Logits --> Out["🎯 logits (B, V)"]

    subgraph Contrastive["🔗 Contrastive Head (InfoNCE)"]
        CAnc["anchor = final_states (B, D)"]
        CPos["positive = item_emb(target) (B, D)"]
        CNeg["negatives = item_emb(hard_neg) (B, N, D)"]
        CLoss["InfoNCE Loss (τ=0.07)"]
        CAnc --> CLoss
        CPos --> CLoss
        CNeg --> CLoss
    end

    FS --> Contrastive
    Out --> Total["L_total = L_CE + 0.3 × L_InfoNCE"]
    CLoss --> Total

    style Input fill:#457B9D,color:#fff,stroke:#333
    style IDEmbed fill:#264653,color:#fff,stroke:#333
    style VisualProj fill:#E76F51,color:#fff,stroke:#333
    style Transformer fill:#2A9D8F,color:#fff,stroke:#333
    style Contrastive fill:#E9C46A,color:#333,stroke:#333
    style Total fill:#F4A261,color:#333,stroke:#333
```

---

## ACT 3 — BRAIN: Multi-Objective Discovery Loss

```mermaid
flowchart TD
    subgraph HeroOutput["🦸 Hero Model Output"]
        L["logits (B, V)"]
        FS["final_states (B, D)"]
    end

    L --> SM["softmax"]
    SM --> SP["softmax_probs\n(B, V)"]

    subgraph Discovery["🔍 Discovery Loss"]
        POP["pop_logit_vector\n(V,) — pre-computed"]
        DOT["dot(softmax_probs,\npop_logit_vector)"]
        PS["per-sample score (B,)"]
        MEAN["mean over batch"]
        LD["L_discovery\n(scalar)"]
        POP --> DOT
        SP --> DOT
        DOT --> PS --> MEAN --> LD
    end

    subgraph CELoss["📊 CE Loss"]
        CE["L_CE(logits, targets)"]
    end

    subgraph CLLoss["🔗 Contrastive Loss"]
        CL["L_contrastive\n(anchor, pos, negs)"]
    end

    L --> CELoss
    FS --> CLLoss

    CE --> Total["🎯 L_total = L_CE\n+ λ_CL × L_contrastive\n+ λ_disc × L_discovery"]
    CL --> Total
    LD --> Total

    subgraph Pareto["⚖️ Pareto Sweep"]
        P1["λ_CL = 0.3 (fixed)"]
        P2["λ_disc ∈ {0.0 ... 1.0}"]
        P3["Optimal: λ=0.3\nnDCG=0.1328\nCoverage=67.6%"]
        P1 --- P2 --- P3
    end

    Total --> Pareto

    style HeroOutput fill:#457B9D,color:#fff,stroke:#333
    style Discovery fill:#E76F51,color:#fff,stroke:#333
    style CELoss fill:#264653,color:#fff,stroke:#333
    style CLLoss fill:#E9C46A,color:#333,stroke:#333
    style Pareto fill:#2A9D8F,color:#fff,stroke:#333
    style Total fill:#F4A261,color:#333,stroke:#333
```
# Differential Expression & Pathway Scoring

## Differential expression analysis

spatioloji_s provides five methods for identifying differentially expressed genes.

### Quick DEG with default method

```python
from spatioloji_s.processing.DEG import run_deg

# All pairwise comparisons
results = run_deg(sp, group_col="cell_type", method="wilcoxon")
```

### Specific comparison

```python
from spatioloji_s.processing.DEG import deg_wilcoxon, deg_ttest

# Wilcoxon rank-sum (non-parametric, recommended default)
results = deg_wilcoxon(sp, group_col="cell_type",
                       groupA="Tumor", groupB="Stroma")

# t-test (parametric)
results = deg_ttest(sp, group_col="cell_type",
                    groupA="Tumor", groupB="Stroma")
```

### Available methods

| Method | Function | Best for |
|--------|----------|----------|
| Wilcoxon | `deg_wilcoxon` | General use, non-parametric, robust |
| t-test | `deg_ttest` | Large sample sizes, normal-ish data |
| MAST | `deg_mast` | Accounting for dropout (zero-inflation) |
| NB-GLM | `deg_nb_glm` | Count data, negative binomial model |
| DESeq2 | `deg_deseq2` | Variance stabilization (requires `pydeseq2`) |

### Results format

All methods return a DataFrame with columns:

| Column | Description |
|--------|-------------|
| `gene` | Gene name |
| `log2fc` | Log2 fold change |
| `pvalue` | Raw p-value |
| `padj` | FDR-adjusted p-value |
| `pct_A` | Percent expressed in group A |
| `pct_B` | Percent expressed in group B |

## Pathway activity scoring

Score gene set activities per cell using decoupler integration.

```python
from spatioloji_s.processing.decoupler import load_gene_sets, make_gene_set_net, score_gene_sets

# Load gene sets (GO, KEGG, MSigDB, etc.)
gene_sets = load_gene_sets(resource="GO:BP")

# Score per cell
score_gene_sets(sp, gene_sets, method="mean", store_key="pathway_scores")

# Access scores in sp.cell_meta or sp.layers
```

### Custom gene sets

```python
custom_sets = {
    "EMT_signature": ["VIM", "CDH2", "SNAI1", "TWIST1", "FN1"],
    "Proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2"],
    "Immune_activation": ["CD8A", "GZMA", "PRF1", "IFNG"],
}

score_gene_sets(sp, custom_sets, method="mean")
```

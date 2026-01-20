# Graph Builder.  
- #### root_dir/src/utils/library_creator.py

This document details the usage, configuration, and data structure required to execute the `library_creator_polars.py` script. This tool automates the construction of molecular graphs (drugs) and genomic graphs (variants) using **PyTorch Geometric**, **Polars**, and **RDKit**.

The script has two main purposes:

1.  **Genomic Module:** Processes variants (SNPs, Indels, Star Alleles) from TSV and VCF files, validates them against a reference genome ``(GRCh38p14)``, and generates directed graphs representing the variant topology.

2.  **Pharmacological Module:** Converts SMILES representations of drugs into molecular graphs with rich atomic and bond features.

```text  
PROJECT/  
├── library_creator_polars.py  
└── data/                       <-- BASE DIR (Mandatory)  
    ├── snp_data_output.tsv     <-- Input: General variants  
    ├── drugs_cid.tsv           <-- Input: Drugs/Compounds  
    ├── ref_genome/             <-- Input: Reference Files  
    │   ├── genome.fna          (Genome FASTA, e.g., GRCh38)  
    │   └── gen_annotations.gff (GFF Annotations - optional)  
    └── haplotype_variants/     <-- Input: PGx variants per gene  
        ├── CYP2D6/  
        │   ├── variant1.vcf  
        │   └── ...  
        ├── DPYD/  
        └── ...
```

This script intends to build graphs for drugs and genes from **variant files** ***(.vcf)*** and compound tables.

## *REQUIREMENTS*

### 1. Libs

* ``polars``: High-performance data processing and vectorized validation.  
* ``torch`` & ``torch_geometric``: Tensor construction and Data objects (graphs).  
* ``rdkit``: Computational chemistry (required for DrugGraphBuilder).  
* ``networkx``: Intermediate graph topology construction (nodes and edges).  
* ``pyfaidx``: Fast indexed access to massive FASTA sequences.  
* ``tqdm``: Progress bar visualization.

### 2. Variants FILE (TSV)

The snp_data_output.tsv file must strictly adhere to the following schema:

| Column | Type | Description |
| :---- | :---- | :---- |
| chr | str | Chromosome identifier (e.g., "1", "X", "chr1"). Must match FASTA headers. |
| start_pos | int | Absolute genomic position (**1-based** format). |
| Ref_Allele | str | Reference allele (must match the FASTA sequence exactly). |
| Alt_Allele | str | Alternative allele. |
| gene | str | (Optional) Associated gene name for grouping output graphs. |
| variant_type | str | (Optional) Explicit type: SNP, MNP, INS, DEL, STAR_ALLELE. |
| FXN_CLASS | str | (Optional) Functional classification (e.g., "missense_variant", "intron_variant"). |

### 3. Drugs FILE (TSV)

The drugs_cid.tsv file is the source for molecular graphs:

| Column | Type | Description |
| :---- | :---- | :---- |
| cid | str/int | Unique compound identifier (e.g., PubChem CID). |
| smiles | str | Valid SMILES string for the compound (e.g., CC(=O)Oc1ccccc1C(=O)O). |
| cmpd_name_cleaned | str | Cleaned drug name (used for naming the .pt output file). |

## ---

**USAGE INSTRUCTIONS**

### **Basic Execution**

The script is designed to run from the project root. It automatically detects the operating system (Debian/Linux or Windows) to handle folder organization scripts.

***Any OS***
```bash
python -m src.utils.library_creator.py

```
### **Command Line Arguments**

The script supports arguments for documentation and testing:

* **Interactive Help:** Displays documentation menus regarding inputs, outputs, and usage.  
  ```bash
  python library_creator_polars.py --help
  ```

* **Verification Mode:** Runs the pipeline but prints specific test logs for a given gene. Useful for validating if a folder in haplotype_variants is being read correctly.  
  ```bash  
  python library_creator_polars.py --verify "CYP2D6"
  ```

### **Outputs**

Files will be generated in src/library/:

1. **genome_library.parquet**: Binary DataFrame containing all validated variants.  
2. **gene_graphs/**: Folders organized by gene containing .pt files (graphs).  
3. **drugs/**: .pt files corresponding to the processed drugs.

## ---

**TROUBLESHOOTING**

| Error / Symptom | Probable Cause | Solution |
| :---- | :---- | :---- |
| **Chr missing: X** | The chromosome name in the TSV does not match the FASTA file header. | Check if your FASTA uses chr1 or NC_000001 nomenclature. The script uses GLOBAL_CHROM_MAPPING to adjust this. |
| **Ref Mismatch** | The reference base in the TSV is not equal to the genome at that position. | Ensure start_pos is **1-based**. The script subtracts 1 internally. Verify you are using the correct genome build (GRCh38 vs hg19). |
| **Invalid POS format** | Non-numeric values in start_pos. | Clean the TSV of "N/A", "-", or dots in position columns. |
| **Missing Drug Graphs** | RDKit failed to parse the SMILES string. | Check the drug_generation_errors.log file generated in the root directory to see which CIDs failed. |
| **Permissions (Linux/Debian)** | Error executing organize_genes.sh. | Ensure you have write permissions in the output folder or manually run chmod +x if the Python script fails to assign permissions. |


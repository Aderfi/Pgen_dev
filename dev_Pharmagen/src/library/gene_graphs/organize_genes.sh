#!/bin/bash
set -e
cd $(dirname "$0")
echo "Organizando grafos por gen..."
mkdir -p UGT1A # Pre-create special case
find . -maxdepth 1 -name "*.pt" -type f | while read filename; do
    base=$(basename "$filename")
    gene_name=$(echo "$base" | cut -d'_' -f1)
    
    # Manejo de casos especiales (genes superpuestos o familias)
    if [[ "$gene_name" =~ ^UGT1A ]]; then 
        mv "$filename" "UGT1A/"
    else
        mkdir -p "$gene_name"
        mv "$filename" "$gene_name/"
    fi
done

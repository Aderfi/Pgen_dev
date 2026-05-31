#!/bin/bash

# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Genome Download Script. Checks version and downloads if updated
# Usage: GDown.sh <genome_name> <version> <output_directory>

URL_passembly="https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
Local_Genome="data/Ref_Genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
LOCAL_GZ="$OUTPUT_DIR/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
LOCAL_FA="$OUTPUT_DIR/HSapiens_GChr38.fa"
INDEX_FAI="$OUTPUT_DIR/HSapiens_GChr38.fa.fai"


echo "Checking reference genome for updates..."

wget --timestamping --directory-prefix="$OUTPUT_DIR" "$URL_passembly"

if [ -f "$LOCAL_GZ" ]; then
    # Re-extract if the .fa is missing or the .gz is newer than the .fa
    if [ ! -f "$LOCAL_FA" ] || [ "$LOCAL_GZ" -nt "$LOCAL_FA" ]; then
        echo "New version detected or extracted FASTA missing."
        echo "Decompressing..."
        # -k keeps the .gz around for future timestamp comparisons
        gunzip -k -f "$LOCAL_GZ"

        echo "Indexing..."

        samtools faidx "$LOCAL_FA"

        echo -e "\n Genome updated and indexed successfully. \n"
        echo -e "\n"
    else
        echo "Local genome is already up to date."
    fi
fi

if [ ! -f "$INDEX_FAI" ]; then
    echo "Index file missing. Indexing..."
    samtools faidx "$LOCAL_FA"
    echo -e "\n Index created successfully. \n"
fi





    
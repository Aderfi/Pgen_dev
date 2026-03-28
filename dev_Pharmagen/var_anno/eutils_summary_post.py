import re
from xml.etree import ElementTree as ET

import pandas as pd
from Bio import Entrez

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
# Es obligatorio para NCBI identificar quién hace la petición.
Entrez.email = "zeooloo@gmail.com"
# Opcional: Obtén una API Key en tu cuenta NCBI para aumentar el límite de 3 a 10 requests/segundo
# Entrez.api_key = "TU_API_KEY"


def get_snp_data_batch(rs_list):
    """
    1. Sube la lista de rsIDs usando epost (History Server).
    2. Descarga los sumarios XML usando esummary y WebEnv.
    3. Parsea el XML con ElementTree para extraer columnas específicas.
    """
    """
    # 1. Limpieza de IDs
    clean_ids = [rs.replace('rs', '').strip() for rs in rs_list if rs]
    if not clean_ids:
        return []
    
    id_list_str = ",".join(clean_ids)
    
    print(f"Procesando {len(clean_ids)} variantes...")
    """
    try:
        """
        # ---------------------------------------------------------------------
        # PASO A: EPOST (Subir IDs al historial del servidor)
        # ---------------------------------------------------------------------
        # Esto es crucial para listas largas. Nos devuelve una 'query_key' y 'WebEnv'.
        post_handle = Entrez.epost(db="snp", id=id_list_str)
        search_results = Entrez.read(post_handle)
        post_handle.close()
        
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"] #

        # ---------------------------------------------------------------------
        # PASO B: ESUMMARY (Descargar datos usando el historial)
        # ---------------------------------------------------------------------
        # Retrievestart=0 y retmax=len(clean_ids) asegura que bajamos todo el lote.
        # Nota: Para listas GIGANTES (miles), deberías hacer un bucle aquí paginando.
        summary_handle = Entrez.esummary(db="snp", query_key=query_key, WebEnv=webenv)
        xml_data = summary_handle.read()
        summary_handle.close()
    """
        # ---------------------------------------------------------------------
        # PASO C: PARSING XML (ElementTree)
        # ---------------------------------------------------------------------

        with open("debug_snp_summary.xml", "r") as f:
            xml_data = f.read()

        root = ET.fromstring(xml_data)

        extracted_data = []

        # En el XML de esummary, cada entrada suele ser un DocumentSummary
        for doc in root.findall(".//DocumentSummary"):
            # --- Extracción Básica ---
            snp_id_val = doc.findtext("SNP_ID")  # El UID numérico
            rs_name = f"rs{snp_id_val}"

            # CHRPOS suele venir como "11:654321" (Cromosoma:Posición)
            chrpos = doc.findtext("CHRPOS")
            if chrpos and ":" in chrpos:
                chr_val, pos_val = chrpos.split(":")
            else:
                chr_val = doc.findtext("CHRONO")  # Backup
                pos_val = "N/A"

            # --- Parsing del campo complejo 'DOCSUM' ---
            # Este campo contiene texto como: "HGVS=NC_...:g.123A>T|SEQ=[A/T]|CLIN=..."
            docsum_text = doc.findtext("DOCSUM", "")
            print(docsum_text)
            # 1. Extraer variante / tipo
            variant_class = doc.findtext("SNP_CLASS")  # Ej: snp, in-del

            # 2. Extraer Alelos (Ref vs Alt)
            # Intentamos buscar patrones HGVS genomic (g.) que indican el cambio real Ref>Alt
            # Regex busca: g.<posicion><Ref>><Alt>
            hgvs_pattern = re.search(r"g\.\d+([ACGT]+)>([ACGT]+)", docsum_text)

            ref_allele = "N/A"
            alt_allele = "N/A"

            if hgvs_pattern:
                ref_allele = hgvs_pattern.group(1)
                alt_allele = hgvs_pattern.group(2)
            else:
                # Fallback: Buscar SEQ=[A/G] (Esto solo nos dice qué alelos existen, no cual es ref seguro)
                seq_pattern = re.search(r"SEQ=\[([ACGT]+)/([ACGT]+)\]", docsum_text)
                if seq_pattern:
                    ref_allele = (
                        seq_pattern.group(1) + "?"
                    )  # Marcamos con ? por incertidumbre
                    alt_allele = seq_pattern.group(2) + "?"

            # 3. Gen
            # A veces viene en tag separado, a veces en GENES dentro de DocSum

            gene_name_nodes = doc.findall(".//GENES/GENE_E/NAME")
            all_genes = [node.text for node in gene_name_nodes if node.text]
            if all_genes:
                # El primero es el gen principal
                gene_val = all_genes[0]

                # Opción A: El resto son alternativos (usando Slicing de listas) -> Más rápido
                # Esto toma todos los genes desde el índice 1 en adelante.
                gene_alt_vals = all_genes[1:]
                gene_alt_vals_str = "|".join(gene_alt_vals)

                # Opción B (Tu lógica original): Excluir explícitamente si se repite el nombre
                # gene_alt_vals = [g for g in all_genes if g != gene_val]

            else:
                gene_val = ""
                gene_alt_vals = []
                gene_alt_vals_str = ""

            # gene_find = doc.find(".//GENES/GENE_E/NAME")
            # gene_val = gene_find.text if gene_find is not None else "__INTERGENIC__"

            # alt_genes_find = doc.findall(".//GENES/GENE_E")
            # gene_alt_vals = [g.find("NAME").text for g in alt_genes_find if g.find("NAME") is not None and g.find("NAME").text != gene_val]

            # 4. Significación Clínica
            # Buscar etiqueta explícita o dentro del texto
            clin_sig = doc.findtext("CLINICAL_SIGNIFICANCE")
            if not clin_sig:
                # Intentar regex en docsum
                clin_pattern = re.search(r"CLIN_SIG=([^|]+)", docsum_text)
                clin_sig = clin_pattern.group(1) if clin_pattern else "N/A"

            variant_seq = (
                re.search(r"SEQ=\[[ACGT/]+\]", docsum_text)[0]
                if re.search(r"SEQ=\[[ACGT/]+\]\|", docsum_text)
                else "N/A"
            )
            variant_len = (
                re.search(r"LEN=\d+", docsum_text)[0]
                if re.search(r"LEN=\d+", docsum_text)
                else "N/A"
            )

            variant_len = (
                variant_len.replace("LEN=", "") if variant_len != "N/A" else variant_len
            )
            variant_seq = (
                variant_seq.replace("SEQ=[", "").replace("]", "")
                if variant_seq != "N/A"
                else variant_seq
            )

            pattern = r"(NC_\d+\.(\d+)):g\.\d+([A-Z])>([A-Z])"
            matches = re.findall(pattern, docsum_text)
            valid_variants = []
            ref_base = None

            grch38_versions = {
                "NC_000001": 11,
                "NC_000002": 12,
                "NC_000003": 12,
                "NC_000004": 12,
                "NC_000005": 10,
                "NC_000006": 12,
                "NC_000007": 14,
                "NC_000008": 11,
                "NC_000009": 12,
                "NC_000010": 11,
                "NC_000011": 10,
                "NC_000012": 12,
                "NC_000013": 11,
                "NC_000014": 9,
                "NC_000015": 10,
                "NC_000016": 10,
                "NC_000017": 11,
                "NC_000018": 10,
                "NC_000019": 10,
                "NC_000020": 11,
                "NC_000021": 9,
                "NC_000022": 11,
                "NC_000023": 11,  # X
                "NC_000024": 10,  # Y
            }

            for full_id, version_str, ref, alt in matches:
                # Separamos NC_000001 y la versión
                base_id = full_id.split(".")[0]
                version = int(version_str)

                # 2. Verificamos si esta versión específica corresponde a GRCh38
                if base_id in grch38_versions and grch38_versions[base_id] == version:
                    # Es una variante GRCh38
                    if ref_base is None:
                        ref_base = ref
                    valid_variants.append(alt)

            unique_alts = sorted(list(set(valid_variants)), key=valid_variants.index)
            if not unique_alts and ref_base:
                unique_alts = (
                    [alt_allele.replace("?", "")] if alt_allele != "N/A" else []
                )
            variant_constructed = (
                f"{ref_base}>{'|'.join(unique_alts)}" if ref_base else "N/A"
            )

            if variant_len.isdigit() and int(variant_len) > 1:
                # Lógica para inserción/deleción
                # Ejemplo SPDI: NC_000005.10:59821462:CTGTCT:CT
                # Ejemplo DOCSUM: HGVS=NC_000005.10:g.59821465_59821468del

                espid = doc.findtext("SPDI", "")
                # Busca patrones SPDI (Chrom:Pos:Ref:Alt)
                spdi_match = re.findall(
                    r"(NC_\d+\.\d+):(\d+):([ACGT]+):([ACGT]+)", espid
                )

                if spdi_match:
                    # Tomamos el cromosoma del primer match encontrado
                    spdi_chrom = spdi_match[0][0]

                    # Construimos la regex dinámica para HGVS
                    # Usamos re.escape para el cromosoma (por los puntos) y añadimos ':g\.' que es estándar en genómica
                    pattern = re.escape(spdi_chrom) + r":g\.(\d+_\d+)(del|dup|ins)"

                    type_variant = re.findall(pattern, docsum_text)

            # Construir fila
            row = {
                "snp": rs_name,
                "snp_id": snp_id_val,
                "chr": chr_val,
                "pos": pos_val,
                "variant": variant_constructed,
                "variant_seq": variant_seq,
                "variant_len": variant_len,
                "variant_type": variant_class,
                "gene": gene_val,
                "gene_alt": gene_alt_vals_str,
                "Ref_Allele": ref_allele,
                "Alt_Allele": alt_allele,
            }
            extracted_data.append(row)

        return extracted_data

    except Exception as e:
        print(f"Ocurrió un error en el proceso: {e}")
        return []


def get_snp_data_batch_nextgen(rs_list):
    """
    Descarga y parsea datos de dbSNP.
    Para variantes complejas (LEN > 1), utiliza SPDI para normalizar
    los alelos a formato tipo VCF (Ref_Seq > Alt_Seq).
    """
    """
    # 1. Limpieza de IDs
    clean_ids = [rs.replace('rs', '').strip() for rs in rs_list if rs]
    if not clean_ids:
        return []
    
    id_list_str = ",".join(clean_ids)
    print(f"Procesando {len(clean_ids)} variantes...")
    """
    try:
        """
        # ---------------------------------------------------------------------
        # PASO A y B: EPOST y ESUMMARY (Entrez)
        # ---------------------------------------------------------------------
        # Recuerda configurar tu email si no lo has hecho fuera de la función:
        # Entrez.email = "tu_email@farmacia.com"
        
        post_handle = Entrez.epost(db="snp", id=id_list_str)
        search_results = Entrez.read(post_handle)
        post_handle.close()
        
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]

        summary_handle = Entrez.esummary(db="snp", query_key=query_key, WebEnv=webenv)
        xml_data = summary_handle.read()
        summary_handle.close()
        """
        # ---------------------------------------------------------------------
        # PASO C: PARSING XML
        # ---------------------------------------------------------------------
        # Para debug local:
        with open("debug_snp_summary.xml", "r") as f:
            xml_data = f.read()

        root = ET.fromstring(xml_data)
        extracted_data = []

        # Mapa de versiones para filtrar solo GRCh38 (Human Build 38)
        grch38_versions = {
            "NC_000001": 11,
            "NC_000002": 12,
            "NC_000003": 12,
            "NC_000004": 12,
            "NC_000005": 10,
            "NC_000006": 12,
            "NC_000007": 14,
            "NC_000008": 11,
            "NC_000009": 12,
            "NC_000010": 11,
            "NC_000011": 10,
            "NC_000012": 12,
            "NC_000013": 11,
            "NC_000014": 9,
            "NC_000015": 10,
            "NC_000016": 10,
            "NC_000017": 11,
            "NC_000018": 10,
            "NC_000019": 10,
            "NC_000020": 11,
            "NC_000021": 9,
            "NC_000022": 11,
            "NC_000023": 11,  # X
            "NC_000024": 10,  # Y
        }

        for doc in root.findall(".//DocumentSummary"):
            # --- Datos Generales ---
            snp_id_val = doc.findtext("SNP_ID")
            rs_name = f"rs{snp_id_val}"
            docsum_text = doc.findtext("DOCSUM", "")

            # Gen
            gene_name_nodes = doc.findall(".//GENES/GENE_E/NAME")
            all_genes = [node.text for node in gene_name_nodes if node.text]
            if all_genes:
                gene_val = all_genes[0]
                gene_alt_vals_str = "|".join(all_genes[1:])
            else:
                gene_val = "__INTERGENIC__"
                gene_alt_vals_str = ""

            # Longitud
            variant_len_str = "0"
            len_match = re.search(r"LEN=(\d+)", docsum_text)
            if len_match:
                variant_len_str = len_match.group(1)
            variant_len_int = int(variant_len_str)

            variant_class = doc.findtext("SNP_CLASS")
            processed_complex = False  # Flag de control

            # =================================================================
            # LÓGICA COMPLEJA: INDELS / DUPS / INS (Normalización VCF)
            # =================================================================
            if variant_len_int > 1:
                espid = doc.findtext("SPDI", "")

                # Regex SPDI: Captura (Chrom):(Pos):(Deleted_Seq):(Inserted_Seq)
                spdi_matches = re.findall(
                    r"(NC_\d+\.\d+):(\d+):([ACGT]*):([ACGT]*)", espid
                )

                if spdi_matches:
                    for spdi_chrom, spdi_pos, spdi_del, spdi_ins in spdi_matches:
                        base_id = spdi_chrom.split(".")[0]
                        version_str = spdi_chrom.split(".")[1]

                        # Filtrar solo GRCh38
                        if (
                            base_id in grch38_versions
                            and int(version_str) == grch38_versions[base_id]
                        ):
                            # Validar con HGVS en DOCSUM para obtener el "Tipo" (del, ins, dup)
                            # Buscamos el cromosoma exacto + patrón g.
                            pattern_hgvs = (
                                re.escape(spdi_chrom) + r":g\.([\d_]+)([a-z]+)"
                            )
                            type_match = re.search(pattern_hgvs, docsum_text)

                            if type_match:
                                processed_complex = True
                                hgvs_pos = type_match.group(1)
                                variant_type_found = type_match.group(2)

                                # --- NORMALIZACIÓN VCF ---
                                # En SPDI:
                                #   Secuencia Borrada = REFERENCIA (REF)
                                #   Secuencia Insertada = ALTERNATIVA (ALT)
                                # Si dbSNP devuelve vacío, usamos un punto '.' (estándar VCF para missing/gap)
                                # o '-' para visualización.

                                vcf_ref = spdi_del if spdi_del else "-"
                                vcf_alt = spdi_ins if spdi_ins else "-"

                                # Construimos la variante visualmente como REF>ALT
                                variant_vcf_str = f"{vcf_ref}>{vcf_alt}"
                                if (
                                    (vcf_ref == vcf_alt[: len(vcf_ref)])
                                    and len(vcf_alt) > len(vcf_ref)
                                    and (vcf_ref in vcf_alt)
                                ):
                                    # Caso de duplicación (ej: REF=CT, ALT=CTCT)
                                    variant_type_found = "dup"

                                row = {
                                    "snp": rs_name,
                                    "snp_id": snp_id_val,
                                    "chr": spdi_chrom,
                                    "pos": spdi_pos,
                                    "hgvs_pos": hgvs_pos,
                                    "variant": variant_vcf_str,  # Formato normalizado
                                    "variant_seq": f"{vcf_ref}/{vcf_alt}",
                                    "variant_len": variant_len_str,
                                    "variant_type": variant_type_found,  # del, ins, dup, inv...
                                    "gene": gene_val,
                                    "gene_alt": gene_alt_vals_str,
                                    "Ref_Allele": vcf_ref,  # Columna REF limpia
                                    "Alt_Allele": vcf_alt,  # Columna ALT limpia
                                }
                                extracted_data.append(row)

            # =================================================================
            # LÓGICA SIMPLE: SNVs O FALLBACK
            # =================================================================
            if not processed_complex:
                chrpos = doc.findtext("CHRPOS")
                if chrpos and ":" in chrpos:
                    chr_val, pos_val = chrpos.split(":")
                else:
                    chr_val = doc.findtext("CHRONO")
                    pos_val = "N/A"

                # Extracción Alelos (Fallback regex)
                hgvs_pattern = re.search(r"g\.\d+([ACGT]+)>([ACGT]+)", docsum_text)
                ref_allele_simple = "N/A"
                alt_allele_simple = "N/A"

                if hgvs_pattern:
                    ref_allele_simple = hgvs_pattern.group(1)
                    alt_allele_simple = hgvs_pattern.group(2)
                else:
                    seq_pattern = re.search(r"SEQ=\[([ACGT]+)/([ACGT]+)\]", docsum_text)
                    if seq_pattern:
                        ref_allele_simple = seq_pattern.group(1)
                        alt_allele_simple = seq_pattern.group(2)

                # Construcción Variante SNV GRCh38
                pattern_snv = r"(NC_\d+\.(\d+)):g\.\d+([A-Z])>([A-Z])"
                matches_snv = re.findall(pattern_snv, docsum_text)

                valid_variants = []
                ref_base = None

                for full_id, version_str, ref, alt in matches_snv:
                    b_id = full_id.split(".")[0]
                    ver = int(version_str)
                    if b_id in grch38_versions and grch38_versions[b_id] == ver:
                        if ref_base is None:
                            ref_base = ref
                        valid_variants.append(alt)

                unique_alts = sorted(list(set(valid_variants)))

                # Si no hubo match HGVS, usar los alelos del regex simple
                if not unique_alts and ref_allele_simple != "N/A":
                    ref_base = ref_allele_simple
                    unique_alts = [alt_allele_simple]

                variant_constructed = (
                    f"{ref_base}>{'|'.join(unique_alts)}" if ref_base else "N/A"
                )

                row = {
                    "snp": rs_name,
                    "snp_id": snp_id_val,
                    "chr": chr_val,
                    "pos": pos_val,
                    "hgvs_pos": "N/A",
                    "variant": variant_constructed,
                    "variant_seq": f"[{ref_allele_simple}/{alt_allele_simple}]",
                    "variant_len": variant_len_str,
                    "variant_type": variant_class,
                    "gene": gene_val,
                    "gene_alt": gene_alt_vals_str,
                    "Ref_Allele": ref_base if ref_base else ref_allele_simple,
                    "Alt_Allele": unique_alts[0] if unique_alts else alt_allele_simple,
                }
                extracted_data.append(row)

        return extracted_data

    except Exception as e:
        print(f"Error procesando batch: {e}")
        return []


def get_snp_data_batch_ULTRA(rs_list):
    """
    Descarga y parsea datos de dbSNP con soporte avanzado para:
    1. Indels/Dups complejos (HGVS range).
    2. Microsatélites/STRs (HGVS brackets [n]).
    3. Normalización a formato VCF (Ref>Alt) basado en SPDI GRCh38.
    """
    """
    clean_ids = [rs.replace('rs', '').strip() for rs in rs_list if rs]
    if not clean_ids:
        return []
    
    id_list_str = ",".join(clean_ids)
    print(f"Procesando {len(clean_ids)} variantes...")
    """
    try:
        """
        # --- PASO A y B: EPOST y ESUMMARY ---
        # Recuerda: Entrez.email = "tu_email@farmacia.com"
        post_handle = Entrez.epost(db="snp", id=id_list_str)
        search_results = Entrez.read(post_handle)
        post_handle.close()
        
        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]

        summary_handle = Entrez.esummary(db="snp", query_key=query_key, WebEnv=webenv)
        xml_data = summary_handle.read()
        summary_handle.close()
        """
        # --- PASO C: PARSING XML ---
        with open("debug_snp_summary.xml", "r") as f:
            xml_data = f.read()

        root = ET.fromstring(xml_data)
        extracted_data = []

        # Mapa de versiones GRCh38
        grch38_versions = {
            "NC_000001": 11,
            "NC_000002": 12,
            "NC_000003": 12,
            "NC_000004": 12,
            "NC_000005": 10,
            "NC_000006": 12,
            "NC_000007": 14,
            "NC_000008": 11,
            "NC_000009": 12,
            "NC_000010": 11,
            "NC_000011": 10,
            "NC_000012": 12,
            "NC_000013": 11,
            "NC_000014": 9,
            "NC_000015": 10,
            "NC_000016": 10,
            "NC_000017": 11,
            "NC_000018": 10,
            "NC_000019": 10,
            "NC_000020": 11,
            "NC_000021": 9,
            "NC_000022": 11,
            "NC_000023": 11,
            "NC_000024": 10,
        }

        for doc in root.findall(".//DocumentSummary"):
            # --- Datos Generales ---
            snp_id_val = doc.findtext("SNP_ID")
            rs_name = f"rs{snp_id_val}"
            docsum_text = doc.findtext("DOCSUM", "")

            # Gen
            gene_name_nodes = doc.findall(".//GENES/GENE_E/NAME")
            all_genes = [node.text for node in gene_name_nodes if node.text]
            if all_genes:
                gene_val = all_genes[0]
                gene_alt_vals_str = "|".join(all_genes[1:])
            else:
                gene_val = "__INTERGENIC__"
                gene_alt_vals_str = ""

            # Longitud
            variant_len_str = "0"
            len_match = re.search(r"LEN=(\d+)", docsum_text)
            if len_match:
                variant_len_str = len_match.group(1)
            variant_len_int = int(variant_len_str)

            variant_class = doc.findtext("SNP_CLASS")
            processed_complex = False

            # =================================================================
            # LÓGICA COMPLEJA: INDELS / DUPS / STRs (LEN > 1)
            # =================================================================
            if variant_len_int > 1:
                espid = doc.findtext("SPDI", "")

                # Regex SPDI: (Chrom):(Pos):(Deleted):(Inserted)
                # spdi_matches es una lista de tuplas [(chrom, pos, del, ins), ...]
                spdi_matches = re.findall(
                    r"(NC_\d+\.\d+):(\d+):([ACGT]*):([ACGT]*)", espid
                )

                if spdi_matches:
                    for spdi_chrom, spdi_pos, spdi_del, spdi_ins in spdi_matches:
                        base_id = spdi_chrom.split(".")[0]
                        version_str = spdi_chrom.split(".")[1]

                        # Filtrar solo GRCh38
                        if (
                            base_id in grch38_versions
                            and int(version_str) == grch38_versions[base_id]
                        ):
                            processed_complex = True

                            # --- ANÁLISIS DEL TIPO DE VARIANTE EN DOCSUM ---
                            # Buscamos coincidencias HGVS asociadas a este cromosoma

                            variant_type_found = "complex"  # Valor por defecto
                            hgvs_pos_str = spdi_pos  # Valor por defecto (posición SPDI)

                            # 1. Regex para Indels/Dups clásicos (Rangos o Puntos + ins/del/dup)
                            # Ej: g.123_125del, g.123dup
                            pattern_indel = (
                                re.escape(spdi_chrom) + r":g\.(\d+)(?:_(\d+))?([a-z]+)"
                            )
                            match_indel = re.search(pattern_indel, docsum_text)

                            # 2. Regex para Microsatélites/STRs (Corchetes [n])
                            # Ej: g.233760235TA[5]
                            # Captura: (start_pos), (secuencia opcional), (numero repeticiones)
                            pattern_str = (
                                re.escape(spdi_chrom) + r":g\.(\d+)([ACGT]*)\[(\d+)\]"
                            )
                            match_str = re.search(pattern_str, docsum_text)

                            if match_str:
                                # Es un STR / Repetición
                                variant_type_found = "microsatellite/STR"
                                hgvs_pos_str = match_str.group(1)  # Start pos
                                # Nota: No extraemos el número de repeticiones para la variante 'vcf'
                                # porque el SPDI ya nos da la secuencia exacta Ref/Alt.

                            elif match_indel:
                                # Es un Indel/Dup/Inv clásico
                                start_pos = match_indel.group(1)
                                end_pos = match_indel.group(
                                    2
                                )  # Puede ser None si es un punto único
                                variant_type_found = match_indel.group(
                                    3
                                )  # del, ins, dup

                                if end_pos:
                                    hgvs_pos_str = f"{start_pos}_{end_pos}"
                                else:
                                    hgvs_pos_str = start_pos

                            # --- CONSTRUCCIÓN DE LA FILA ---
                            # Usamos SPDI para la definición exacta de nucleótidos (VCF style)
                            vcf_ref = spdi_del if spdi_del else "-"
                            vcf_alt = spdi_ins if spdi_ins else "-"

                            variant_vcf_str = f"{vcf_ref}>{vcf_alt}"

                            row = {
                                "snp": rs_name,
                                "snp_id": snp_id_val,
                                "chr": spdi_chrom,
                                "pos": spdi_pos,  # Posición exacta 0-based del SPDI
                                "start_pos_hgvs": hgvs_pos_str,  # Posición/Rango visual del HGVS
                                "variant": variant_vcf_str,
                                "variant_seq": f"{vcf_ref}/{vcf_alt}",
                                "variant_len": variant_len_str,
                                "variant_type": variant_type_found,
                                "gene": gene_val,
                                "gene_alt": gene_alt_vals_str,
                                "Ref_Allele": vcf_ref,
                                "Alt_Allele": vcf_alt,
                            }
                            extracted_data.append(row)

            # =================================================================
            # LÓGICA SIMPLE: SNVs (Fallback)
            # =================================================================
            if not processed_complex:
                # ... (El código para SNVs simples se mantiene igual que la versión anterior)
                # Recuperar lógica simple si es necesario o dejar vacía si solo te interesan complejos
                # Copio la lógica SNV para completitud:
                chrpos = doc.findtext("CHRPOS")
                chr_val, pos_val = (
                    chrpos.split(":")
                    if chrpos and ":" in chrpos
                    else (doc.findtext("CHRONO"), "N/A")
                )

                hgvs_pattern = re.search(r"g\.\d+([ACGT]+)>([ACGT]+)", docsum_text)
                ref_simple = hgvs_pattern.group(1) if hgvs_pattern else "N/A"
                alt_simple = hgvs_pattern.group(2) if hgvs_pattern else "N/A"

                if ref_simple == "N/A":
                    seq_pat = re.search(r"SEQ=\[([ACGT]+)/([ACGT]+)\]", docsum_text)
                    if seq_pat:
                        ref_simple, alt_simple = seq_pat.group(1), seq_pat.group(2)

                pattern_snv = r"(NC_\d+\.(\d+)):g\.\d+([A-Z])>([A-Z])"
                matches_snv = re.findall(pattern_snv, docsum_text)
                valid_alts = []
                ref_base = None

                for full_id, ver, r, a in matches_snv:
                    base = full_id.split(".")[0]
                    if base in grch38_versions and int(ver) == grch38_versions[base]:
                        if ref_base is None:
                            ref_base = r
                        valid_alts.append(a)

                unique_alts = sorted(list(set(valid_alts)))
                if not unique_alts and ref_simple != "N/A":
                    ref_base = ref_simple
                    unique_alts = [alt_simple]

                variant_constructed = (
                    f"{ref_base}>{'|'.join(unique_alts)}" if ref_base else "N/A"
                )

                row = {
                    "snp": rs_name,
                    "snp_id": snp_id_val,
                    "chr": chr_val,
                    "pos": pos_val,
                    "start_pos_hgvs": pos_val,
                    "variant": variant_constructed,
                    "variant_seq": f"[{ref_simple}/{alt_simple}]",
                    "variant_len": variant_len_str,
                    "variant_type": variant_class,
                    "gene": gene_val,
                    "gene_alt": gene_alt_vals_str,
                    "Ref_Allele": ref_base if ref_base else ref_simple,
                    "Alt_Allele": unique_alts[0] if unique_alts else alt_simple,
                }
                extracted_data.append(row)

        return extracted_data

    except Exception as e:
        print(f"Error procesando batch: {e}")
        return []


def get_snp_data_batch_FINAL(rs_list):
    clean_ids = [rs.replace("rs", "").strip() for rs in rs_list if rs]
    if not clean_ids:
        return []

    id_list_str = ",".join(clean_ids)
    print(f"Procesando {len(clean_ids)} variantes...")

    try:
        # --- EPOST y ESUMMARY ---
        # Entrez.email = "zeooloo@gmail.com"
        post_handle = Entrez.epost(db="snp", id=id_list_str)
        search_results = Entrez.read(post_handle)
        post_handle.close()

        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]

        summary_handle = Entrez.esummary(db="snp", query_key=query_key, WebEnv=webenv)
        xml_data = summary_handle.read()
        summary_handle.close()

        # --- PASO C: PARSING XML ---
        # with open('debug_snp_summary.xml', 'r') as f:
        #    xml_data = f.read()
        # --- PARSING ---
        root = ET.fromstring(xml_data)
        extracted_data = []

        grch38_versions = {
            "NC_000001": 11,
            "NC_000002": 12,
            "NC_000003": 12,
            "NC_000004": 12,
            "NC_000005": 10,
            "NC_000006": 12,
            "NC_000007": 14,
            "NC_000008": 11,
            "NC_000009": 12,
            "NC_000010": 11,
            "NC_000011": 10,
            "NC_000012": 12,
            "NC_000013": 11,
            "NC_000014": 9,
            "NC_000015": 10,
            "NC_000016": 10,
            "NC_000017": 11,
            "NC_000018": 10,
            "NC_000019": 10,
            "NC_000020": 11,
            "NC_000021": 9,
            "NC_000022": 11,
            "NC_000023": 11,
            "NC_000024": 10,
        }

        for doc in root.findall(".//DocumentSummary"):
            # --- Datos Generales ---
            snp_id_val = doc.findtext("SNP_ID")
            rs_name = f"rs{snp_id_val}"
            docsum_text = doc.findtext("DOCSUM", "")

            # Gen
            gene_name_nodes = doc.findall(".//GENES/GENE_E/NAME")
            all_genes = [node.text for node in gene_name_nodes if node.text]
            if all_genes:
                gene_val = all_genes[0]
                gene_alt_vals_str = "|".join(all_genes[1:])
            else:
                gene_val = ""
                gene_alt_vals_str = ""

            # Longitud
            variant_len_str = "0"
            len_match = re.search(r"LEN=(\d+)", docsum_text)
            if len_match:
                variant_len_str = len_match.group(1)
            variant_len_int = int(variant_len_str)

            variant_class = doc.findtext("SNP_CLASS")
            processed_complex = False

            fxn_class_val = doc.findtext("FXN_CLASS", "N/A")

            # =================================================================
            # LÓGICA COMPLEJA (LEN > 1)
            # =================================================================
            if variant_len_int > 1:
                espid = doc.findtext("SPDI", "")
                spdi_matches = re.findall(
                    r"(NC_\d+\.\d+):(\d+):([ACGT]*):([ACGT]*)", espid
                )

                if spdi_matches:
                    for spdi_chrom, spdi_pos, spdi_del, spdi_ins in spdi_matches:
                        base_id = spdi_chrom.split(".")[0]
                        version_str = spdi_chrom.split(".")[1]

                        if (
                            base_id in grch38_versions
                            and int(version_str) == grch38_versions[base_id]
                        ):
                            processed_complex = True

                            # --- Extracción de Tipo (Previa) ---
                            variant_type_found = "complex"
                            hgvs_pos_str = spdi_pos

                            pattern_indel = (
                                re.escape(spdi_chrom) + r":g\.(\d+)(?:_(\d+))?([a-z]+)"
                            )
                            match_indel = re.search(pattern_indel, docsum_text)
                            pattern_str = (
                                re.escape(spdi_chrom) + r":g\.(\d+)([ACGT]*)\[(\d+)\]"
                            )
                            match_str = re.search(pattern_str, docsum_text)

                            if match_str:
                                variant_type_found = "microsatellite/STR"
                                hgvs_pos_str = match_str.group(1)
                            elif match_indel:
                                start_pos = match_indel.group(1)
                                end_pos = match_indel.group(2)
                                variant_type_found = match_indel.group(3)
                                hgvs_pos_str = (
                                    f"{start_pos}_{end_pos}" if end_pos else start_pos
                                )

                            # --- DEFINICIÓN ALELOS (SPDI -> VCF) ---
                            vcf_ref = spdi_del if spdi_del else "-"
                            vcf_alt = spdi_ins if spdi_ins else "-"

                            # --- CORRECCIÓN DE TIPO BASADA EN LONGITUD (NUEVO) ---
                            # Si la longitud física contradice la etiqueta de texto, la corregimos.
                            len_ref = len(spdi_del)
                            len_alt = len(spdi_ins)

                            if len_alt > len_ref:
                                # Es inserción o duplicación neta, aunque el texto diga 'del'
                                variant_type_found = "ins/dup"
                            elif len_ref > len_alt:
                                # Es deleción neta
                                variant_type_found = "del"

                            # --- CÁLCULO DE COORDENADAS (NUEVO) ---
                            try:
                                v_start = int(spdi_pos)
                                v_end = v_start + (variant_len_int - 1)
                            except ValueError:
                                v_start = spdi_pos
                                v_end = "N/A"

                            # --- NORMALIZACIÓN NOMBRE CROMOSOMA INT --- (RE SUB FINAL)
                            # Convertir NC_000001.11 a 1, NC_000023.11 a X, etc.
                            norm_chrom = (
                                (spdi_chrom.split(".")[0])
                                .replace("NC_00000", "")
                                .replace("NC_0000", "")
                            )
                            if norm_chrom == "23":
                                norm_chrom = "X"
                            elif norm_chrom == "24":
                                norm_chrom = "Y"
                            spdi_chrom = norm_chrom

                            row = {
                                "snp": rs_name,
                                "snp_id": snp_id_val,
                                "chr": spdi_chrom,
                                "start_pos": v_start,  # Renombrado
                                "end_pos": v_end,  # Nueva columna calculada
                                "start_pos_hgvs": hgvs_pos_str,
                                "variant": f"{vcf_ref}>{vcf_alt}",
                                "variant_seq": f"{vcf_ref}/{vcf_alt}",
                                "variant_len": variant_len_str,
                                "variant_type": variant_type_found,  # Tipo corregido
                                "gene": gene_val,
                                "gene_alt": gene_alt_vals_str,
                                "Ref_Allele": vcf_ref,
                                "Alt_Allele": vcf_alt,
                                "FXN_CLASS": fxn_class_val,
                            }
                            extracted_data.append(row)

            # =================================================================
            # LÓGICA SIMPLE: SNVs
            # =================================================================
            if not processed_complex:
                chrpos = doc.findtext("CHRPOS")
                chr_val, pos_val = (
                    chrpos.split(":")
                    if chrpos and ":" in chrpos
                    else (doc.findtext("CHRONO"), "N/A")
                )

                # Cálculo Coordenadas Simple
                try:
                    v_start_simple = int(pos_val)
                    # Si es SNV (len 1), end = start + (1-1) = start
                    v_end_simple = v_start_simple + (variant_len_int - 1)
                except ValueError:
                    v_start_simple = pos_val
                    v_end_simple = "N/A"

                hgvs_pattern = re.search(r"g\.\d+([ACGT]+)>([ACGT]+)", docsum_text)
                ref_simple = hgvs_pattern.group(1) if hgvs_pattern else "N/A"
                alt_simple = hgvs_pattern.group(2) if hgvs_pattern else "N/A"

                if ref_simple == "N/A":
                    seq_pat = re.search(r"SEQ=\[([ACGT]+)/([ACGT]+)\]", docsum_text)
                    if seq_pat:
                        ref_simple, alt_simple = seq_pat.group(1), seq_pat.group(2)

                pattern_snv = r"(NC_\d+\.(\d+)):g\.\d+([A-Z])>([A-Z])"
                matches_snv = re.findall(pattern_snv, docsum_text)
                valid_alts = []
                ref_base = None

                for full_id, ver, r, a in matches_snv:
                    base = full_id.split(".")[0]
                    if base in grch38_versions and int(ver) == grch38_versions[base]:
                        if ref_base is None:
                            ref_base = r
                        valid_alts.append(a)

                unique_alts = sorted(list(set(valid_alts)))
                if not unique_alts and ref_simple != "N/A":
                    ref_base = ref_simple
                    unique_alts = [alt_simple]

                variant_constructed = (
                    f"{ref_base}>{'|'.join(unique_alts)}" if ref_base else "N/A"
                )

                row = {
                    "snp": rs_name,
                    "snp_id": snp_id_val,
                    "chr": chr_val,
                    "start_pos": v_start_simple,
                    "end_pos": v_end_simple,
                    "start_pos_hgvs": pos_val,
                    "variant": variant_constructed,
                    "variant_seq": f"[{ref_simple}/{alt_simple}]",
                    "variant_len": variant_len_str,
                    "variant_type": variant_class,
                    "gene": gene_val,
                    "gene_alt": gene_alt_vals_str,
                    "Ref_Allele": ref_base if ref_base else ref_simple,
                    "Alt_Allele": unique_alts[0] if unique_alts else alt_simple,
                    "FXN_CLASS": fxn_class_val,
                }
                extracted_data.append(row)

        return extracted_data

    except Exception as e:
        print(f"Error procesando batch: {e}")
        return []


def main(RS_FILE, OUTPUT_FILE):
    with open(RS_FILE, "r") as f:
        my_rsids = [line.strip() for line in f.readlines() if line.strip()]

    data = pd.DataFrame(get_snp_data_batch_FINAL(my_rsids))
    print(data)

    data.to_csv(OUTPUT_FILE, sep="\t", index=False)


if __name__ == "__main__":
    RS_FILE = "all_snps_list.txt"
    OUTPUT_FILE = "snp_data_output.tsv"
    main(RS_FILE, OUTPUT_FILE)
    print("=" * 50)
    print("\n\tDonete")

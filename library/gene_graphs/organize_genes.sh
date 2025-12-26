#!/bin/bash

# Mensaje de inicio
echo "Iniciando organización de archivos..."

# Contador para feedback visual (opcional, pero útil con 80k archivos)
count=0

# Iterar sobre todos los archivos .pt en el directorio actual
for filename in *.pt; do
    # Verificar que el archivo existe (evita errores si no hay coincidencias)
    [ -e "$filename" ] || continue

    # ---------------------------------------------------------
    # PASO 1: Extraer el nombre del gen base
    # ---------------------------------------------------------
    # Sintaxis ${variable%%_*} elimina todo desde el PRIMER guion bajo hacia el final.
    # Ejemplo: SCL;SOA5;GGT_variant.pt  --> SCL;SOA5;GGT
    raw_gene="${filename%%_*}"

    # ---------------------------------------------------------
    # PASO 2: Manejo de punto y coma (;)
    # ---------------------------------------------------------
    # Sintaxis ${variable%%;*} elimina todo desde el PRIMER punto y coma hacia el final.
    # Si no hay ';', la cadena se queda igual.
    # Ejemplo: SCL;SOA5;GGT --> SCL
    gene_name="${raw_gene%%;*}"

    # ---------------------------------------------------------
    # PASO 3: Caso Especial UGT1Ax (1-10)
    # ---------------------------------------------------------
    # Verificamos con Regex si es UGT1A seguido de un dígito (1-9) o el número 10.
    # Si coincide, forzamos el directorio a "UGT1A".
    if [[ "$gene_name" =~ ^UGT1A([1-9]|10)$ ]]; then
        target_dir="UGT1A"
    else
        target_dir="$gene_name"
    fi

    # ---------------------------------------------------------
    # PASO 4: Mover el archivo
    # ---------------------------------------------------------
    # Crear el directorio si no existe (-p lo hace silencioso si ya existe)
    mkdir -p "$target_dir"
    
    # Mover el archivo dentro
    mv "$filename" "$target_dir/"

    # Feedback simple cada 1000 archivos para saber que no se ha colgado
    ((count++))
    if (( count % 1000 == 0 )); then
        echo "Procesados $count archivos..."
    fi

done

echo "Proceso finalizado. Total archivos movidos: $count"

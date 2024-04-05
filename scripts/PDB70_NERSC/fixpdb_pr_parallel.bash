#!/bin/bash

PROGNAME=$0

usage() {
  cat << EOF >&2
Usage: $PROGNAME -f <file> -o <dir>

-f <file>: the input PDB file path
-o  <dir>: the output dir
EOF
 exit 1
}

file=""
outdir=""

while getopts f:o: options
do
  case $options in
    (f) file=$OPTARG;;
    (o) outdir=$OPTARG;;
    (*) usage
  esac
done

if [ "$file" == "" ]
then
  echo "Parameter -f must be provided"
  exit 1
fi

if [ "$outdir" == "" ]
then
  echo "Parameter -o must be provided"
  exit 1
fi

pdb_loc=${outdir}/pdb
seq_loc=${outdir}/sequence
pr_loc=${outdir}/saxs_r
ref_seq_loc=${outdir}/ref_seq

#mkdir -p ${pdb_loc}
#echo "created PDB storage folder at ${pdb_loc}"
#mkdir -p ${seq_loc}
#echo "created sequence storage folder at ${seq_loc}"
#mkdir -p ${pr_loc}
#echo "created saxs P(r) storage folder at ${pr_loc}"

if [ -f "$file" ]; then
   done=`grep -c ${file} ${outdir}/complete.txt`
   echo $done
   if [ "$done" -ne 0 ] ; then
       exit
   fi
   echo ${file}
   
   filename=$(basename "$file")
   prefix="${filename%.*}"
   IFS='_' read -r PDB_code chain_ID <<< "$prefix"
   echo ${filename}
   
   python get_full_seq.py -f ${file} -o ${ref_seq_loc}
   if [ ! -f "${ref_seq_loc}/fixed_${prefix}.fasta" ]; then
   	echo "BASH_ref_seq_Error: File ${ref_seq_loc}/fixed_${prefix}.fasta does not exist."
        exit
   fi
   pdbfixer ${file} --output "${pdb_loc}/fixed_${filename}" \
   --keep-heterogens=none --replace-nonstandard --add-residues --add-atoms=all  --verbose \
   --sequence "${ref_seq_loc}/fixed_${prefix}.fasta" --chain_id "${chain_ID}"
   if [ ! -f "${pdb_loc}/fixed_${filename}" ]; then
        echo "BASH_fixer_Error: File ${pdb_loc}/fixed_${filename} does not exist."
        exit
   fi
   calc-pr "${pdb_loc}/fixed_${filename}" -o "${pr_loc}/${filename}.pr.csv"
   extract-seq -f "${pdb_loc}/fixed_${filename}" -o "${seq_loc}/${filename}.fasta"
   echo ${file} >> ${outdir}/complete.txt
fi

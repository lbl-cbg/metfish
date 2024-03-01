#!/bin/bash

PROGNAME=$0

usage() {
  cat << EOF >&2
Usage: $PROGNAME -f <file> -o <dir>

-f <file>: the input file containing a list of pdb files (can be obtained using ls > file_name.txt)
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
   done=`grep -c ${file} complete.txt`
   echo $done
   if [ "$done" -ne "0" ] ; then
       exit
   fi
   echo ${file}
   
   filename=$(basename "$file")
   prefix="${filename%.*}"
   IFS='_' read -r PDB_code chain_ID <<< "$prefix"
   echo ${filename}
   
   python get_full_seq.py -f ${file} -o ${ref_seq_loc}
   pdbfixer ${file} --output "${pdb_loc}/fixed_${filename}" \
   --keep-heterogens=none --replace-nonstandard --add-residues --add-atoms=all  --verbose \
   --sequence "${ref_seq_loc}/fixed_${prefix}.fasta" --chain_id "${chain_ID}"
   calc-pr "${pdb_loc}/fixed_${filename}" -o "${pr_loc}/${filename}.pr.csv"
   python PDBtoSeq.py  -f "${pdb_loc}/fixed_${filename}" -o "${seq_loc}/${filename}.fasta" 
   echo ${file} >> complete.txt
fi

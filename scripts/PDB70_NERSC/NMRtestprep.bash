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
msa_loc=${outdir}/msa

mkdir -p ${pdb_loc}
echo "created PDB storage folder at ${pdb_loc}"
mkdir -p ${seq_loc}
echo "created sequence storage folder at ${seq_loc}"
mkdir -p ${pr_loc}
echo "created saxs P(r) storage folder at ${pr_loc}"
mkdir -p ${msa_loc}
echo "created MSA storage folder at ${msa_loc}"

if [ -f "$file" ]; then
   done=`grep -c ${file} ${outdir}/complete.txt`
   echo $done
   if [ "$done" -ne 0 ] ; then
       exit
   fi
   echo ${file}
   
   filename=$(basename "$file")
   prefix="${filename%.*}"
   IFS='_' read -r PDB_code chain_ID frame_number<<< "$prefix"
   echo ${filename}
   
   python "$SCRATCH/metfish/scripts/PDB70_NERSC/PDBtotest_NMR.py" -f ${file} -o ${seq_loc}

   calc-pr "${pdb_loc}/${filename}" -o "${pr_loc}/${prefix}.pr.csv"

   #Calc_MSA

   echo ${file} >> ${outdir}/complete.txt
fi

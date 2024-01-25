#!/bin/bash
# Author: Feng Yu fyu2@lbl.gov 2024

# CoDNaS is a protein conformational database based of entire proteins as derived from PDB. 
# For each represented protein, the database contains the redundant collection of all corresponding different structures. 

# This script is used to calculate SAXS curve for PDBs based on lists of PDBs downloading from the CoDNaS server.
# This script will generate SAXS curve, P(r) curve , extract sequences and download the PDBs for each PDB entry.



PROGNAME=$0

usage() {
  cat << EOF >&2
Usage: $PROGNAME -f <file> -o <dir>

-f <file>: the input file containing a list of pdb files downloaded from CoDNaS
-o  <dir>: the output dir
EOF
 exit 1
}

filename=""
outdir=""

while getopts f:o: options
do
  case $options in 
    (f) filename=$OPTARG;;
    (o) outdir=$OPTARG;;
    (*) usage
  esac
done

if [ "$filename" == "" ]
then 
  echo "Parameter -f must be provided"
  exit 1
fi

if [ "$outdir" == "" ]
then
  echo "Parameter -o must be provided"
  exit 1
else  
  echo $outdir
  awk 'NR>1 { print substr($1,1,4)}' ${filename} | sort -u |tr '\n' ' ' | sed '$s/ $/\n/'| tr ' ' ',' > ${outdir}/clean_pdb.output
fi

pdb_loc=${outdir}/pdb
seq_loc=${outdir}/sequence
saxs_loc=${outdir}/saxs_q
pr_loc=${outdir}/saxs_r

mkdir ${pdb_loc}
echo "created PDB storage folder at ${pdb_loc}"
mkdir ${seq_loc}
echo "created sequence storage folder at ${seq_loc}"

./batch_download_seq.sh -f ${outdir}/clean_pdb.output  -o ${outdir} -p -q

mkdir ${saxs_loc}
echo "created SAXS measurement storage folder at ${saxs_loc}"
mkdir ${pr_loc}
echo "created P(r) storage folder at ${pr_loc}"

full_pdb_list=$(cat ${outdir}/clean_pdb.output)

IFS=',' read -ra pdbs <<< "$full_pdb_list"

for pdb in ${pdbs[@]}
do
 if [ ! -f ${pdb_loc}/${pdb}.pdb.gz ]
 then
   echo no pdb file for $pdb
 else
   gunzip -c ${pdb_loc}/${pdb}.pdb.gz > ${pdb_loc}/${pdb}.pdb
   rm ${pdb_loc}/${pdb}.pdb.gz
   foxs ${pdb_loc}/${pdb}.pdb 
   mv ${pdb_loc}/${pdb}.pdb.dat ${saxs_loc}/${pdb}.dat
   python SAXS_to_pr.py -f ${saxs_loc}/${pdb}.dat -o ${pr_loc}
 fi
done

rm ${outdir}/clean_pdb.output


#!/bin/bash

# Script to download files from RCSB http file download services.
# Use the -h switch to get help on usage.

if ! command -v curl &> /dev/null
then
    echo "'curl' could not be found. You need to install 'curl' for this script to work."
    exit 1
fi

PROGNAME=$0
BASE_URL="https://files.rcsb.org/download"
BASE_URL_SEQ="https://www.rcsb.org/fasta/entry"

usage() {
  cat << EOF >&2
Usage: $PROGNAME -f <file> [-o <dir>] [-c] [-p]
 modified from RCSB script
 -f <file>: the input file containing a comma-separated list of PDB ids
 -o  <dir>: the output dir, default: current dir
 -c       : download a cif.gz file for each PDB id and store in the pdb subfolder
 -p       : download a pdb.gz file for each PDB id (not available for large structures) and store in the pdb subfolder
 -q       : download a .fasta file for each PDB id (sequence) and sotre in the sequence subfolder
EOF
  exit 1
}

download() {
  url="$BASE_URL/$1"
  out=$2/$1
  echo "Downloading $url to $out"
  curl -s -f $url -o $out || echo "Failed to download $url"
}

download_seq() {
  url="$BASE_URL_SEQ/$1"
  out="$2/$1.fasta"
  echo "Downloading $url to $out" 
  curl -s -f $url -o $out || echo "Failed to download $url"
}

listfile=""
outdir="."
cif=false
pdb=false
seq=false
while getopts f:o:cpq op
do
  case $op in
    (f) listfile=$OPTARG;;
    (o) outdir=$OPTARG;;
    (c) cif=true;;
    (p) pdb=true;;
    (q) seq=true;;   
    (*) usage
  esac
done
shift "$((OPTIND - 1))"

if [ "$listfile" == "" ]
then
  echo "Parameter -f must be provided"
  exit 1
fi
contents=$(cat $listfile)

# see https://stackoverflow.com/questions/918886/how-do-i-split-a-string-on-a-delimiter-in-bash#tab-top
IFS=',' read -ra tokens <<< "$contents"

for token in "${tokens[@]}"
do
  if [ "$cif" == true ]
  then
    download ${token}.cif.gz ${outdir}/pdb
  fi
  if [ "$pdb" == true ]
  then
    download ${token}.pdb.gz ${outdir}/pdb
  fi
  if [ "$seq" == true ]
  then 
    download_seq ${token} ${outdir}/sequence
  fi
done









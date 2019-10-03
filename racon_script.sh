#!/bin/bash 
#

RACON=/home/aakdemir/racon/build/bin/racon
#00-Albacore 01-Albacore2 02-basecRAWller 03-Chiron 04-Metrichor
raw_read=${1}
out_folder=${2}
if [ ! -d "${out_folder}" ]; then
    echo "ARDA"
    mkdir ${out_folder}
    mkdir ${out_folder}"/racon_assembly"
    mkdir ${out_folder}"/racon_assembly/consensus"
fi
#mkdir ${out_folder}
out_file="${out_folder}/racon_assembly/consensus/"
echo $out_file
if [ ! -f "${out_file}merge_1_par.fastq" ]; then
    cp ${raw_read} ${out_file}"merge_1_par.fastq"
fi
#cd out_file
#racon_dir

for i in 1
do
    minimap2/minimap2 -x ava-ont -k12 -w5  ${out_file}merge_${i}_par.fastq ${out_file}merge_${i}_par.fastq > ${out_file}reads_${i}.paf
    miniasm/miniasm   -f   ${out_file}merge_${i}_par.fastq ${out_file}reads_${i}.paf > ${out_file}raw_contigs_${i}.gfa
    awk '$1 ~/S/ {print ">"$2"\n"$3}' ${out_file}raw_contigs_${i}.gfa > ${out_file}raw_contigs_${i}.fasta
    echo "Running minimap with raw_contigs and merge_1_par.fastq"
	minimap2/minimap2   ${out_file}raw_contigs_${i}.fasta ${out_file}merge_${i}_par.fastq > ${out_file}mapping_${i}.paf
    echo "Racon mapping"
	${RACON}   ${out_file}merge_${i}_par.fastq ${out_file}mapping_${i}.paf ${out_file}raw_contigs_${i}.fasta > ${out_file}consensus_${i}_0.fasta
    ###racon loop
    for j in 0 1 2 3 4 5 6 7 8 9 10
    #for j in 1 3 4 5 6 7 8 9 10
    #for j in `seq 20 49`
    do
        minimap2/minimap2  ${out_file}consensus_${i}_${j}.fasta ${out_file}merge_${i}_par.fastq > ${out_file}map${i}_${j}.paf
        ${RACON}   ${out_file}merge_${i}_par.fastq ${out_file}map${i}_${j}.paf  ${out_file}consensus_${i}_${j}.fasta >  ${out_file}consensus_${i}_$((j+1)).fasta

    done

done


#SBATCH --job-name=consensus
#SBATCH --output=cons
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G # mb
#









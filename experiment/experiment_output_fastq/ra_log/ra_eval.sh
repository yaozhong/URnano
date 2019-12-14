#!/bin/sh


ecoli_ref="/home/yaozhong/working/2_nanopore/data/chiron_fast5_arda/ecoli.fasta"
lambda_ref="/home/yaozhong/working/2_nanopore/data/chiron_fast5_arda/lambda.fasta"
human_chr11_ref="/home/yaozhong/working/2_nanopore/data/human_test_20190623/chr11/reference/chr11.fa"

GRAPHMAP="/home/yaozhong/working/2_nanopore/software_tools/graphmap/bin/Linux-x64/graphmap"
SOFT=" /yshare3/home/yaozhong/.usr/local/bin/jsa.hts.errorAnalysis"


echo "=======================Chiron============================="
echo "-----------------------[Ecloi]----------------------------"
out_fold="./chiron"
$GRAPHMAP align -r $ecoli_ref -d $out_fold/chiron_ecoli.fastq -o $out_fold/chiron_ecoli.bam
$SOFT -bamFile=$out_fold/chiron_ecoli.bam -reference=$ecoli_ref


echo "-----------------------[Phage]----------------------------"
$GRAPHMAP align -r $lambda_ref -d $out_fold/chiron_phage.fastq -o $out_fold/chiron_phage.bam
$SOFT -bamFile=$out_fold/chiron_phage.bam -reference=$lambda_ref


echo "-----------------------[Human chr11]----------------------------"
$GRAPHMAP align -r $human_chr11_ref -d $out_fold/chiron_human_chr11.fastq -o $out_fold/chiron_human_chr11.bam
$SOFT -bamFile=$out_fold/chiron_human_chr11.bam -reference=$human_chr11_ref


echo "=======================Guppy Taiyaki============================="
echo "-----------------------[Ecloi]----------------------------"
out_fold="./taiyaki"
$GRAPHMAP align -r $ecoli_ref -d $out_fold/guppy_ecoli.fasta -o $out_fold/guppy_ecoli.bam
$SOFT -bamFile=$out_fold/guppy_ecoli.bam -reference=$ecoli_ref


echo "-----------------------[Phage]----------------------------"
$GRAPHMAP align -r $lambda_ref -d $out_fold/guppy_phage.fasta -o $out_fold/guppy_phage.bam
$SOFT -bamFile=$out_fold/guppy_phage.bam -reference=$lambda_ref


echo "-----------------------[Human chr11]----------------------------"
$GRAPHMAP align -r $human_chr11_ref -d $out_fold/guppy_human_chr11.fasta -o $out_fold/guppy_human_chr11.bam
$SOFT -bamFile=$out_fold/guppy_human_chr11.bam -reference=$human_chr11_ref



echo "=======================URnano============================="
echo "-----------------------[Ecloi]----------------------------"
out_fold="./urnano"
$GRAPHMAP align -r $ecoli_ref -d $out_fold/ur_ecoli.fastq -o $out_fold/ur_ecoli.bam
$SOFT -bamFile=$out_fold/ur_ecoli.bam -reference=$ecoli_ref


echo "-----------------------[Phage]----------------------------"
$GRAPHMAP align -r $lambda_ref -d $out_fold/ur_phage.fastq -o $out_fold/ur_phage.bam
$SOFT -bamFile=$out_fold/ur_phage.bam -reference=$lambda_ref


echo "-----------------------[Human chr11]----------------------------"
$GRAPHMAP align -r $human_chr11_ref -d $out_fold/ur_human_chr11.fastq -o $out_fold/ur_human_chr11.bam
$SOFT -bamFile=$out_fold/ur_human_chr11.bam -reference=$human_chr11_ref








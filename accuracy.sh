#!/bin/bash
### get reference and a raw read first align using graphmap
ref_path=$1
raw_reads=$2
#echo $ref_path
Graphmap=/home/aakdemir/graphmap/bin/Linux-x64/graphmap
Japsa=.usr/local/bin/jsa.hts.errorAnalysis
name="${raw_reads%%.fastq}"
filenameind=$(echo $name | grep -b -o "/" | tail -1)
#echo $filenameind
ind=${filenameind%%:*}
filename=${name:(ind+1)}
#echo $filename
outfile=${filename}_errorAnalysis.txt
echo "Outputting to: "${filename}.sam
${Graphmap} align -r ${ref_path} -d ${raw_reads} -o $filename.sam
echo "Aligned ${raw_reads} to ${ref_path}"
${Japsa} --bamFile=${filename}.sam --reference=${ref_path} > ${outfile}
cat ${outfile}
echo "Results stored in: "${outfile}

#!/bin/bash
raw=${1}
out=${2}
ref=${3}
bash racon_script.sh ${raw} ${out}
python script/assembly_assess.py ${out} ${ref} 0

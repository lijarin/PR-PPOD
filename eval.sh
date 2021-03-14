
#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Documents

weights=(50 100 150 200 870 990)

for weight in ${weights[@]};do
	#echo $weight
	python -u eval.py --trained_model /share/splend/adversiral_defense/ssd-fed.pytorch/weights/2-3client/ssd-${weight}.pth >> result_${weight}.txt

done

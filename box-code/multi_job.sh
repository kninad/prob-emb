#!/bin/bash

COUNTER=0
trn_folder="exp2.3_baseline_notaxo"
for w1 in 1.0; do
for w2 in 0.1; do
for learning_rate in 0.01 0.001; do 
for r1 in 0.5 0.1; do 
for embed_dim in 5 20 50; do
echo "hyper-params: ${w1},${w2},${learning_rate},${r1},${embed_dim}"
COUNTER=$((COUNTER + 1))
runFile="run_${trn_folder}_${COUNTER}.sh"
touch ${runFile}
echo "#!/bin/bash" > ${runFile}
echo "source activate tfcpu-py27" >> ${runFile}
echo "python experiments/trainer.py \\" >> ${runFile}
echo "--train_dir='./data/book_data/${trn_folder}/' \\" >> ${runFile}
echo "--r1=${r1} --w1=${w1} --w2=${w2} \\" >> ${runFile}
echo "--learning_rate=${learning_rate} --embed_dim=${embed_dim} \\" >> ${runFile}
echo "--max_steps=100000 --batch_size=512 --print_every=1000 --useLossKL=True" >> ${runFile}
echo "source deactivate tfcpu-py27" >> ${runFile}
#cat ${runFile}
sbatch --partition titanx-short --mem 30000 --gres=gpu:1 ${runFile}
echo ""
echo "Starting Next Run"
echo "**********************"
done
done
done
done
done

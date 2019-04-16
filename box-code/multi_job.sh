#!/bin/bash

### ADD embedding dim too  
for w1 in 1.0; do
for w2 in 0.1; do
for learning_rate in 0.01 0.001; do 
for r1 in 0.5 0.1; do 
for embed_dim in 5 20 50; do
echo "hyper-params: ${w1},${w2},${learning_rate},${r1},${embed_dim}"
touch run.sh
echo "#!/bin/bash" > run.sh
echo "source activate tfcpu-py27" >> run.sh
echo "python experiments/trainer.py \\" >> run.sh
echo "--train_dir='./data/book_data/exp2.3_baseline_notaxo/' \\" >> run.sh
echo "--r1=${r1} --w1=${w1} --w2=${w2} \\" >> run.sh
echo "--learning_rate=${learning_rate} --embed_dim=${embed_dim} \\" >> run.sh
echo "--max_steps=100000 --batch_size=512 --print_every=1000 --useLossKL=True" >> run.sh
echo "source deactivate tfcpu-py27" >> run.sh
sbatch --partition titanx-short --mem 30000 --gres=gpu:1 run.sh
echo ""
echo "Starting Next Run"
echo "**********************"
done
done
done
done
done

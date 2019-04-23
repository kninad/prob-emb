#!/bin/sh

COUNTER=0
trn_folder="exp2.2_int_taxo"    # for use in exp_name
trn_file="exp2.2_int_taxo/book_train_hb.csv"    # for dset arg param 

#for dim in 30; do # emd dim
#for lr in 0.1; do # learning_rate
#done
#done

dim=30
lr=0.1

epochs=3
negs=50
burnin=5
train_threads=4
gpu=-1

echo "Submitting Job"
echo "************************"
COUNTER=$((COUNTER + 1))

exp_name="$(date "+%Y-%m-%d_%H-%M-%S")_${trn_folder}_dim${dim}_lr${lr}_epoc${epochs}_neg${negs}_burn${burnin}"

echo "Exp name: ${exp_name}"

logpath="./log/${exp_name}/"
chkpath="./log/${exp_name}/book_model.pth"
mkdir ${logpath}

# Call the python script
#source activate poincare
python3 embed.py \
       -dim ${dim} \
       -lr ${lr} \
       -epochs ${epochs} \
       -negs ${negs} \
       -burnin ${burnin} \
       -gpu ${gpu} \
       -dset "./data/book_data/${trn_file}" \
       -logfolder ${logpath} \
       -checkpoint ${chkpath} \
       -fresh \
       -sparse \
       -manifold poincare
#source deactivate

#runFile="run_${trn_folder}_${COUNTER}.sh"
#touch ${runFile}

#echo "#!/bin/bash" > ${runFile}
#echo "source activate tfcpu-py27" >> ${runFile}
#echo "python experiments/trainer.py \\" >> ${runFile}
#echo "--train_dir='./data/book_data/${trn_folder}/' \\" >> ${runFile}
#echo "--r1=${r1} --w1=${w1} --w2=${w2} \\" >> ${runFile}
#echo "--learning_rate=${learning_rate} --embed_dim=${embed_dim} \\" >> ${runFile}

#echo "--train_file='genre_genre_master.txt' --train_test_file='genre_genre_eval.txt' \\" 
#echo "--max_steps=1500 --batch_size=512 --print_every=1000 --useLossKL=True" >> ${runFile}

## echo "--init_embedding='pre_train' \\" >> ${runFile}
## echo "--max_steps=100000 --batch_size=512 --print_every=1000 --useLossKL=True" >> ${runFile}

##echo "source deactivate tfcpu-py27" >> ${runFile}
##cat ${runFile}
#sbatch --partition titanx-short --mem 30000 --gres=gpu:1 ${runFile}
#echo ""





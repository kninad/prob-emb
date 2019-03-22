#!/bin/bash

#Set the unmerged data dir
#rootdir="/home/ninad/Desktop/Link-to-sem4/dsis/prob-emb/box-code/data/book_data/book_data_4.5_500_taxonomy/"
rootdir=$1

name_margs="${rootdir}unmerged_data/book_marginal_prob.txt"
name_vocab="${rootdir}unmerged_data/book_vocabulary.txt"

genre_margs="${rootdir}unmerged_data/genre_marginal_prob.txt"
genre_vocab="${rootdir}unmerged_data/genre_vocabulary.txt"

output_margs="${rootdir}book_marginal_prob.txt"
output_vocab="${rootdir}vocabulary.txt"

cat $name_margs $genre_margs > $output_margs
cat $name_vocab $genre_vocab > $output_vocab


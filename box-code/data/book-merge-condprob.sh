#!/bin/bash

#Set the unmerged data dir
#rootdir="/home/ninad/Desktop/Link-to-sem4/dsis/prob-emb/box-code/data/book_data/book_data_4.5_500_taxonomy/"
rootdir=$1

mv_mv_trn="${rootdir}unmerged_data/book_book_master.txt_train.txt"
mv_mv_dev="${rootdir}unmerged_data/book_book_master.txt_dev.txt"
mv_mv_tst="${rootdir}unmerged_data/book_book_master.txt_tst.txt"

mv_gn_trn="${rootdir}unmerged_data/book_genre_master.txt_train.txt"
mv_gn_dev="${rootdir}unmerged_data/book_genre_master.txt_dev.txt"
mv_gn_tst="${rootdir}unmerged_data/book_genre_master.txt_tst.txt"

gn_gn_trn="${rootdir}unmerged_data/genre_genre_master.txt_train.txt"
gn_gn_dev="${rootdir}unmerged_data/genre_genre_master.txt_dev.txt"
gn_gn_tst="${rootdir}unmerged_data/genre_genre_master.txt_tst.txt"

output_trn="${rootdir}book_train.txt"
output_dev="${rootdir}book_dev.txt"
output_tst="${rootdir}book_test.txt"

cat $mv_mv_trn $mv_gn_trn $gn_gn_trn > $output_trn
cat $mv_mv_dev $mv_gn_dev $gn_gn_dev > $output_dev
cat $mv_mv_tst $mv_gn_tst $gn_gn_tst > $output_tst

# Also create the rel.txt file if its not present
rel="${rootdir}rel.txt"
echo "isa" > $rel


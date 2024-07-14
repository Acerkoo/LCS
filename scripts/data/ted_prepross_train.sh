export CUDA_VISIBLE_DEVICES=0

repo_dir=../dataset/ted
raw_dir=$repo_dir/raw

# train spm model
# echo "============ step: train spm model =========="
# << M
vocab_dir=./vocab
model_prefix=$vocab_dir/spm_64k
vocab_size=64000
character_coverage=0.9995
input_sentence_size=1000000

train_data=./train.txt
cat $raw_dir/supervised/*/train.* >> $train_data

fairseq_code=../../fairseq/fairseq-main
script_code=$fairseq_code/scripts/

if [ ! -d $vocab_dir ]; then
    mkdir -p $vocab_dir
fi


token_dir=./token_data
python $script_code/spm_train.py --input=$train_data --model_prefix=$model_prefix --vocab_size=$vocab_size --character_coverage=$character_coverage --input_sentence_size=$input_sentence_size --num_threads=64 | exit 1;

# echo "spm train done."

# <<M
# # data tokenize
if [ ! -d $token_dir ]; then
    mkdir -p $token_dir
fi
train_src=./train.src
train_tgt=./train.tgt
valid_src=./valid.src
valid_tgt=./valid.tgt
test_src=./test.src
test_tgt=./test.tgt

echo "============ step: data tokenize =========="
for dataset in $( ls $raw_dir/supervised/ ); do
    IFS='-' read -r -a langs <<< $dataset
    l1=${langs[0]}
    l2=${langs[1]}

    for lang in ${langs[@]}; do
        for split in train valid; do
            python $script_code/spm_encode.py --model ${vocab_dir}/spm_64k.model --input $raw_dir/supervised/$dataset/$split.$dataset.$lang --outputs $token_dir/$split.$dataset.$lang
        done
    done

    awk '{print "<'${l1}'> "$0}' $token_dir/train.$dataset.${l1} >> $train_src
    awk '{print "<'${l2}'> "$0}' $token_dir/train.$dataset.${l2} >> $train_tgt
    awk '{print "<'${l2}'> "$0}' $token_dir/train.$dataset.${l2} >> $train_src
    awk '{print "<'${l1}'> "$0}' $token_dir/train.$dataset.${l1} >> $train_tgt

    awk '{print "<'${l1}'> "$0}' $token_dir/valid.$dataset.$l1 | head -n 200 >> $test_src
    awk '{print "<'${l2}'> "$0}' $token_dir/valid.$dataset.$l2 | head -n 200 >> $test_tgt
    awk '{print "<'${l2}'> "$0}' $token_dir/valid.$dataset.$l2 | head -n 200 >> $test_src
    awk '{print "<'${l1}'> "$0}' $token_dir/valid.$dataset.$l1 | head -n 200 >> $test_tgt
done
# M
for dataset in $( ls $raw_dir/zero-shot/ ); do
    IFS='-' read -r -a langs <<< $dataset
    l1=${langs[0]}
    l2=${langs[1]}
    for lang in ${langs[@]}; do
        for split in test; do
            python $script_code/spm_encode.py --model ${vocab_dir}/spm_64k.model --input $raw_dir/zero-shot/$dataset/test.$dataset.$lang --outputs $token_dir/test.$dataset.$lang
        done
    done
    awk '{print "<'${l1}'> "$0}' $token_dir/test.$dataset.$l1 | head -n 20 >> $valid_src
    awk '{print "<'${l2}'> "$0}' $token_dir/test.$dataset.$l2 | head -n 20 >> $valid_tgt
    awk '{print "<'${l2}'> "$0}' $token_dir/test.$dataset.$l2 | head -n 20 >> $valid_src
    awk '{print "<'${l1}'> "$0}' $token_dir/test.$dataset.$l1 | head -n 20 >> $valid_tgt
done
echo "data tokenize done."

echo "============ step: update dict =========="
dict=./data-bin/dict.txt
mkdir -p ./data-bin
cut -f 1 $vocab_dir/spm_64k.vocab | tail -n +4 | sed "s/$/ 100/g" > $dict | exit 1;
echo "<en> 100" >> $dict
for dataset in $( ls $raw_dir/supervised/ ); do
    IFS='-' read -r -a langs <<< $dataset
    l1=${langs[0]}
    l2=${langs[1]}
    if [ $l1 != 'en' ]; then
        echo "<$l1> 100" >> $dict
    else 
        echo "<$l2> 100" >> $dict
    fi
done
echo "update dict done."
M

echo "============ step: data binarize =========="
dict=./data-bin/dict.txt
fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref ./train \
    --validpref ./valid \
    --testpref ./test \
    --destdir ./data-bin \
    --workers 64 \
    --srcdict $dict \
    --tgtdict $dict
echo "data binarize done."

# M
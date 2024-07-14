export CUDA_VISIBLE_DEVICES=0

raw_dir=../dataset/ted/raw

# train spm model
# echo "============ step: train spm model =========="
vocab_dir=./vocab
model_prefix=$vocab_dir/spm_64k

fairseq_code=../../fairseq/fairseq-main
script_code=$fairseq_code/scripts/

# if [ ! -d $vocab_dir ]; then
#     mkdir -p $vocab_dir
# fi

# echo "spm train done."

# # data tokenize
token_dir=./token_data
test_dir=./test

# <<M
echo "============ step: data tokenize =========="
for dataset in $( ls $raw_dir/supervised/ ); do
    IFS='-' read -r -a langs <<< $dataset
    l1=${langs[0]}
    l2=${langs[1]}
    for lang in ${langs[@]}; do
        for split in test; do
            python $script_code/spm_encode.py --model ${vocab_dir}/spm_64k.model --input $raw_dir/supervised/$dataset/$split.$dataset.$lang --outputs $token_dir/$split.$dataset.$lang
        done
    done
    output_dir=$test_dir/supervised/$l1-$l2
    if [ ! -d $output_dir ]; then
        mkdir -p $output_dir
    fi
    awk '{print "<'${l1}'> "$0}' $token_dir/test.$dataset.$l1 > $output_dir/test.$l1
    awk '{print "<'${l2}'> "$0}' $token_dir/test.$dataset.$l2 > $output_dir/test.$l2
    cat $raw_dir/supervised/$dataset/test.$dataset.$l2 > $output_dir/ref.$l2

    l1=${langs[1]}
    l2=${langs[0]}
    output_dir=$test_dir/supervised/$l1-$l2
    if [ ! -d $output_dir ]; then
        mkdir -p $output_dir
    fi
    awk '{print "<'${l1}'> "$0}' $token_dir/test.$dataset.$l1 > $output_dir/test.$l1
    awk '{print "<'${l2}'> "$0}' $token_dir/test.$dataset.$l2 > $output_dir/test.$l2
    cat $raw_dir/supervised/$dataset/test.$dataset.$l2 > $output_dir/ref.$l2
done
# M
for dataset in $( ls $raw_dir/zero-shot/ ); do
    IFS='-' read -r -a langs <<< $dataset
    l1=${langs[0]}
    l2=${langs[1]} 
    output_dir=$test_dir/zero-shot/$l1-$l2
    if [ ! -d $output_dir ]; then
        mkdir -p $output_dir
    fi
    
    for lang in ${langs[@]}; do
        python $script_code/spm_encode.py --model ${vocab_dir}/spm_64k.model --input $raw_dir/zero-shot/$dataset/test.$dataset.$lang --outputs $token_dir/test.$dataset.$lang
    done

    echo $token_dir/test.$dataset.$l1
    awk '{print "<'${l1}'> "$0}' $token_dir/test.$dataset.$l1 > $output_dir/test.$l1
    awk '{print "<'${l2}'> "$0}' $token_dir/test.$dataset.$l2 > $output_dir/test.$l2
    cat $raw_dir/zero-shot/$dataset/test.$dataset.$l2 > $output_dir/ref.$l2

    l1=${langs[1]}
    l2=${langs[0]} 
    output_dir=$test_dir/zero-shot/$l1-$l2
    if [ ! -d $output_dir ]; then
        mkdir -p $output_dir
    fi
    awk '{print "<'${l1}'> "$0}' $token_dir/test.$dataset.$l1 | head -n 2000 > $output_dir/test.$l1
    awk '{print "<'${l2}'> "$0}' $token_dir/test.$dataset.$l2 | head -n 2000 > $output_dir/test.$l2
    cat $raw_dir/zero-shot/$dataset/test.$dataset.$l2 | head -n 2000 > $output_dir/ref.$l2
done
echo "data tokenize done."


test_fair=./test-bin
dict=./data-bin/dict.txt

# <<M
echo "============ step: data binarize =========="
for dataset in $( ls $test_dir/supervised/ ); do
    IFS='-' read -r -a langs <<< $dataset
    l1=${langs[0]}
    l2=${langs[1]}
    output_dir=$test_fair/supervised/$l1-$l2
    fairseq-preprocess --source-lang $l1 --target-lang $l2 \
        --testpref $test_dir/supervised/$dataset/test \
        --destdir $output_dir \
        --workers 64 \
        --srcdict $dict \
        --tgtdict $dict
done
# M
for dataset in $( ls $test_dir/zero-shot/ ); do
    IFS='-' read -r -a langs <<< $dataset
    l1=${langs[0]}
    l2=${langs[1]}
    output_dir=$test_fair/zero-shot/$l1-$l2
    fairseq-preprocess --source-lang $l1 --target-lang $l2 \
        --testpref $test_dir/zero-shot/$dataset/test \
        --destdir $output_dir \
        --workers 64 \
        --srcdict $dict \
        --tgtdict $dict
done
echo "data binarize done."
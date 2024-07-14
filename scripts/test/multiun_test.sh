repo_dir=$( cd $( dirname $0 )/../.. && pwd )

LC=../tools/mosesdecoder/scripts/tokenizer/lowercase.perl
DETOK=../tools/mosesdecoder/scripts/tokenizer/detokenizer.perl

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

tag_style=$1
fairseq_branch=$2 
signature=$3 # model strategy
ckpt=$4
avg=$5

test=$repo_dir/data/multiun/$tag_style/test
code_dir=$repo_dir/fairseq/fairseq-$fairseq_branch

bleu() {
    SRCS=$1
    for langpair in "${SRCS[@]}"; do
        # echo $SRC >> tmp/gen.$MODEL.out
        IFS='-' read -r -a langs <<< $langpair
        src=${langs[0]}
        tgt=${langs[1]}
        echo $src '->' $tgt
        echo $src '->' $tgt >> $output
        test_dir=$test/${src}-${tgt}/data-bin

        extra_args="--infer-style t-enc"
        if [[ $tag_style =~ "t_dec" ]]; then
            extra_args="--infer-style s-enc-t-dec"
        fi

        touch $result_dir/gen.${src}-${tgt}.out 
        python $code_dir/fairseq_cli/generate.py $test_dir --path $model_check --beam 5 \
            --task translation_label --lang-prefix-tok "<$tgt>" \
            $extra_args \
            --remove-bpe  --batch-size 128 | grep -E '[S|H|P|T]-[0-9]+' > $result_dir/gen.${src}-${tgt}.out
        grep -E '[H]-[0-9]+' < $result_dir/gen.${src}-${tgt}.out > $result_dir/gen.${src}-${tgt}.sys
        
        cat $result_dir/gen.${src}-${tgt}.sys | sort -n -k 2 -t - | cut -f3- | perl $LC | sed -e "s/<$tgt> //g" | perl $DETOK -l $tgt > $result_dir/gen.${src}-${tgt}.detok.sys
        ref_file=$repo_dir/data/multiun/raw/testset/UNv1.0.testset.$tgt.ref
        if [ ! -f $ref_file ]; then
            cat $repo_dir/data/multiun/raw/testset/UNv1.0.testset.$tgt | perl $LC > $ref_file
        fi

        if [ $tgt == "zh" ] ; then 
            bleuscore=$( cat $result_dir/gen.${src}-${tgt}.detok.sys | sacrebleu $ref_file --tokenize zh )
        else 
            bleuscore=$( cat $result_dir/gen.${src}-${tgt}.detok.sys | sacrebleu $ref_file )
        fi

        bleustr="$(date "+%Y-%m-%d %H:%M:%S")\t${model_check}\t${src}-${tgt}\nsacrebleu: ${bleuscore}"
        echo -e $bleustr
        echo -e $bleustr >> $output
        echo '------------'
        echo '------------' >> $output
        # exit 0;
    done 
}

pt_branch=checkpoint_${ckpt}.pt
model_check=$repo_dir/checkpoints/multiun/$signature/$avg/$pt_branch
result_dir=$repo_dir/results/multiun/$signature/$pt_branch
output=$result_dir/result.out

if [ ! -f $model_check ]; then
    echo $model_check "is not exists."
    exit 0;
fi

echo $result_dir
echo $model_check

if [ ! -d $result_dir ]; then
    mkdir -p $result_dir
    touch $output
    chmod 777 $result_dir -R
fi

# supervised translation
SRCS=(
    "ar-en"
    "en-ar"
    "zh-en"
    "en-zh"
    "ru-en"
    "en-ru"
)
echo "=============== supervised test ================="
echo "=============== supervised test =================" >> $output
bleu $SRCS

# zero-shot directions
SRCS=(
    "ru-ar"
    "ar-ru"
    "zh-ar"
    "ar-zh"
    "zh-ru"
    "ru-zh"
)

echo "=============== zero-shot test ================="
echo "=============== zero-shot test =================" >> $output
bleu $SRCS
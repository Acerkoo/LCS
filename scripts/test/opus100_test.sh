repo_dir=$( cd $( dirname $0 )/../.. && pwd )

tag_style=$1
fairseq_branch=$2 
signature=$3 # model strategy
ckpt=$4
avg=$5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

test=$repo_dir/data/opus100/$tag_style/test-bin
ref_dir=$repo_dir/data/opus100/$tag_style/test
spm_model=$repo_dir/data/opus100/$tag_style/vocab/spm_64k.model
code_dir=$repo_dir/fairseq/fairseq-$fairseq_branch

echo "test_dir:" $test
echo "fairseq_code:" $code_dir

bleu() {
    src=$1
    tgt=$2
    cls=$3
    # echo $src $tgt $cls
    echo $src '->' $tgt
    echo $src '->' $tgt >> $output
    dataset=${src}-${tgt}
    
    test_dir=$test/$cls/${src}-${tgt}

    extra_args="--infer-style t-enc"
    if [[ $tag_style =~ "t_dec" ]]; then
        extra_args="--infer-style s-enc-t-dec"
    fi

    touch $result_dir/gen.${src}-${tgt}.out 
    python $code_dir/fairseq_cli/generate.py $test_dir --path $model_check --beam 5 --lenpen 1.0 \
        --task translation_label --lang-prefix-tok "<$tgt>" \
        $extra_args \
        --remove-bpe --batch-size 128  | grep -E '[S|H|P|T]-[0-9]+' > $result_dir/gen.${src}-${tgt}.out
    cat $result_dir/gen.${src}-${tgt}.out | grep -E '[H]-[0-9]+' | sort -n -k 2 -t - | cut -f3- | sed -e "s/<$tgt> //g" > $result_dir/gen.${src}-${tgt}.sys
    
    python $code_dir/scripts/spm_decode.py --model=${spm_model} --input=$result_dir/gen.${src}-${tgt}.sys --input_format=piece > $result_dir/gen.${src}-${tgt}.detok.sys

    ref=$ref_dir/$cls/$dataset/ref.$tgt

    if [ $tgt == "zh" ] ; then 
        bleuscore=$( cat $result_dir/gen.${dataset}.detok.sys | sacrebleu $ref --tokenize zh )
    else 
        bleuscore=$( cat $result_dir/gen.${dataset}.detok.sys | sacrebleu $ref )
    fi

    bleustr="$(date "+%Y-%m-%d %H:%M:%S")\t${model_check}\t${src}-${tgt}\nsacrebleu: ${bleuscore}"
    echo -e $bleustr
    echo -e $bleustr >> $output
    echo '------------'
    echo '------------' >> $output
}

test() {
    srcs=$1
    IFS=',' read -r -a langlist <<< $srcs
    for lang in ${langlist[@]}; do
        bleu $lang en supervised
        bleu en $lang supervised
    done
}
test2() {
    srcs=$1
    IFS=',' read -r -a langpairs <<< $srcs
    for dataset in ${langpairs[@]}; do
        IFS='-' read -r -a langs <<< $dataset
        l1=${langs[0]}
        l2=${langs[1]}
        bleu $l1 $l2 zero-shot
        bleu $l2 $l1 zero-shot
    done
}

pt_branch=checkpoint_${ckpt}.pt
model_check=$repo_dir/checkpoints/opus100/$signature/$avg/$pt_branch
result_dir=$repo_dir/results/opus100/$signature/$pt_branch
output=$result_dir/result.out

if [ ! -f $model_check ]; then
    echo "no model" $model_check
    exit 0;
fi

echo $result_dir
echo $model_check

if [ ! -d $result_dir ]; then
    mkdir -p $result_dir
    touch $output
    chmod 777 $result_dir -R
fi

high_pairs="ar,bg,bn,bs,ca,cs,da,de,el,es,et,eu,fa,fi,fr,he,hr,hu,id,is,it,ja,ko,lt,lv,mk,ms,mt,nl,no,pl,pt,ro,ru,si,sk,sl,sq,sr,sv,th,tr,uk,vi,zh"
low_pairs="am,be,fy,gd,ha,ig,kk,kn,ky,li,mr,my,oc,or,ps,se,te,tk,ug,yi,zu"
mid_pairs="af,as,az,br,cy,eo,ga,gl,gu,hi,ka,km,ku,mg,ml,nb,ne,nn,pa,rw,sh,ta,tg,tt,ur,uz,wa,xh"
zero_shot="ar-de,ar-fr,ar-nl,ar-ru,ar-zh,de-fr,de-nl,de-ru,de-zh,fr-nl,fr-ru,fr-zh,nl-ru,nl-zh,ru-zh"

echo "=============== Zero-shot test ================="
echo "=============== Zero-shot test =================" >> $output
test2 ${zero_shot}

echo "=============== High test ================="
echo "=============== High test =================" >> $output
test $high_pairs

echo "=============== Mid test ================="
echo "=============== Mid test =================" >> $output
test $mid_pairs

echo "=============== Low test ================="
echo "=============== Low test =================" >> $output
test $low_pairs
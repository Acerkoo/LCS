cur_dir=$( cd $( dirname $0 ) && pwd )
repo_dir=$( cd $cur_dir/../../.. && pwd )

signature=$1
step=$2
result_dir=$repo_dir/results/opus100/$signature/checkpoint_$step.pt

acc() {
    srcs=$1
    IFS=',' read -r -a datasets <<< $srcs
    for dataset in ${datasets[@]}; do
        IFS='-' read -r -a langs <<< $dataset
        src=${langs[0]}
        tgt=${langs[1]}
        gen=$result_dir/gen.${dataset}.detok.sys
        echo $gen $tgt
        res=$( python $cur_dir/detect.py $gen $src $tgt )
        echo $dataset
        echo $res
        echo '----------------'
    done
}

zeroshot="ar-de,ar-fr,ar-nl,ar-ru,ar-zh,de-ar,de-fr,de-nl,de-ru,de-zh,fr-ar,fr-de,fr-nl,fr-ru,fr-zh,nl-ar,nl-de,nl-fr,nl-ru,nl-zh,ru-ar,ru-de,ru-fr,ru-nl,ru-zh,zh-ar,zh-de,zh-fr,zh-nl,zh-ru"
acc $zeroshot
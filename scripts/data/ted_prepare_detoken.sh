detokenizer=../tools/mosesdecoder/scripts/tokenizer/detokenizer.perl

src_dict=./data
tgt_dict=./raw

for dataset in $( ls $src_dict/supervised/ ); do
    IFS='-' read -r -a langs <<< $dataset
    mkdir -p $tgt_dict/supervised/$dataset/
    for split in train valid test; do
        for lang in ${langs[@]}; do
            if [ $lang =~ 'pt' ]; then
                lang='pt'
            elif [ $lang =~ 'zh' ]; then
                lang='zh'
            fi
            if [ ! -f $tgt_dict/supervised/$dataset/$split.$dataset.$lang ]; then
                echo $split.$dataset.$lang
                perl $detokenizer -q -u -a -l $lang < $src_dict/supervised/$dataset/$split.$dataset.$lang > $tgt_dict/supervised/$dataset/$split.$dataset.$lang 
            fi
        done
    done
done

for dataset in $( ls $src_dict/zero-shot/ ); do
    IFS='-' read -r -a langs <<< $dataset
    mkdir -p $tgt_dict/zero-shot/$dataset/
    for lang in ${langs[@]}; do
        if [ ! -f $tgt_dict/zero-shot/$dataset/test.$dataset.$lang ]; then
            echo "generate "$tgt_dict/zero-shot/$dataset/test.$dataset.$lang
            perl $detokenizer -q -u -a -l $lang < $src_dict/zero-shot/$dataset/test.$dataset.$lang > $tgt_dict/zero-shot/$dataset/test.$dataset.$lang
        fi
    done
done

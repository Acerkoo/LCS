prep=./
tmp=$prep/tmp
DICT=$prep/data-bin/dict.src.txt

TEST=$prep/test
token_data=../tokenized_data

prepro() {
    SRCS=$1
    echo "process test data..."
    for SRC in "${SRCS[@]}"; do
        array=(${SRC//-/ })
        src=${array[0]}
        tgt=${array[1]}
        echo $src "->" $tgt
        test_dir=$TEST/${src}-${tgt}
        mkdir -p $test_dir

        awk '{print "<'$src'> "$0}' $tmp/test.bpe.$src > $test_dir/test.$SRC.$src
        awk '{print "<'$tgt'> "$0}' $tmp/test.bpe.$tgt > $test_dir/test.$SRC.$tgt
        # break
        fairseq-preprocess --source-lang $src --target-lang $tgt \
            --testpref $test_dir/test.$SRC \
            --srcdict $DICT --tgtdict $DICT \
            --destdir $test_dir/data-bin/ \
            --workers 64
    done
}

SRCS=(
    "ar-en"
    "en-ar"
    "zh-en"
    "en-zh"
    "ru-en"
    "en-ru"
)
echo "supervised test ..."
prepro $SRCS

# zero-shot directions
SRCS=(
    "ru-ar"
    "ar-ru"
    "zh-ar"
    "ar-zh"
    "zh-ru"
    "ru-zh"
)
echo "zero-shot test ..."
prepro $SRCS
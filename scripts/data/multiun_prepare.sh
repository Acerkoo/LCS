data_path=../dataset/multiun
tmp=../dataset/multiun/raw

paste -d'|' $datapath/en-zh/UNv1.0.en-zh.en $datapath/en-zh/UNv1.0.en-zh.zh | cat -n | shuf -n 2000000 | sort -n | cut -f2 > $datapath/en-zh/sample
cut -d'|' -f1 $datapath/en-zh/sample > $tmp/multiun.zh-en.en
cut -d'|' -f2 $datapath/en-zh/sample > $tmp/multiun.zh-en.zh

paste -d'|' $datapath/en-ru/UNv1.0.en-ru.en $datapath/en-ru/UNv1.0.en-ru.ru | cat -n | shuf -n 2000000 | sort -n | cut -f2 > $datapath/en-ru/sample
cut -d'|' -f1 $datapath/en-ru/sample > $tmp/multiun.ru-en.en
cut -d'|' -f2 $datapath/en-ru/sample > $tmp/multiun.ru-en.ru

paste -d'|' $datapath/ar-en/UNv1.0.ar-en.en $datapath/ar-en/UNv1.0.ar-en.ar | cat -n | shuf -n 2000000 | sort -n | cut -f2 > $datapath/ar-en/sample
cut -d'|' -f1 $datapath/ar-en/sample > $tmp/multiun.ar-en.en
cut -d'|' -f2 $datapath/ar-en/sample > $tmp/multiun.ar-en.ar

SRCS=(
    "ar-en"
    "ru-en"
)

PROC_SCRIPTS=../tools/mosesdecoder/scripts
TOKENIZER=$PROC_SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$PROC_SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$PROC_SCRIPTS/tokenizer/normalize-punctuation.perl
LC=$PROC_SCRIPTS/tokenizer/lowercase.perl
REM_NON_PRINT_CHAR=$PROC_SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

echo "pre-processing train data..."
for SRC in "${SRCS[@]}"; do
    echo $SRC
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]}
    
    for LANG in ${array[@]}; do
        echo $LANG
        cat $tmp/multiun.$SRC.$LANG | \
            perl $NORM_PUNC $LANG | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $LANG > $tmp/multiun.tokenized.$SRC.$LANG
    done
    
done

# zh-en
python -m jieba $tmp/multiun.zh-en.zh -d > $tmp/multiun.tokenized.zh-en.zh

cat $tmp/multiun.zh-en.en | \
    perl $NORM_PUNC en | \
    perl $REM_NON_PRINT_CHAR | \
    perl $TOKENIZER -threads 8 -a -l en > $tmp/multiun.tokenized.zh-en.en
prep=.
token_data=../dataset/multiun
tmp=$prep/tmp

if [ ! -d $tmp ]; then
    mkdir -p $tmp
fi

PROC_SCRIPTS=../tools/mosesdecoder/scripts
TOKENIZER=$PROC_SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$PROC_SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$PROC_SCRIPTS/tokenizer/normalize-punctuation.perl
LC=$PROC_SCRIPTS/tokenizer/lowercase.perl
REM_NON_PRINT_CHAR=$PROC_SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=40000

TRAIN=$prep/train.mnmt
BPE_CODE=$prep/code
# rm $TRAIN

echo $BPE_CODE
echo $token_data
# <<M
cat $token_data/multiun.tokenized.* >> $TRAIN

echo "learn_bpe.py on ${TRAIN}..."
subword-nmt learn-bpe -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for l in ar zh ru; do 
    subword-nmt apply-bpe -c $BPE_CODE < $token_data/multiun.tokenized.${l}-en.${l} > $tmp/multiun.bpe.tokenized.${l}-en.${l}
    subword-nmt apply-bpe -c $BPE_CODE < $token_data/multiun.tokenized.${l}-en.en > $tmp/multiun.bpe.tokenized.${l}-en.en

    awk '{print "<'$l'> "$0}' $tmp/multiun.bpe.tokenized.${l}-en.${l} >> $prep/train.src
    awk '{print "<en> "$0}' $tmp/multiun.bpe.tokenized.${l}-en.en >> $prep/train.tgt
    awk '{print "<en> "$0}' $tmp/multiun.bpe.tokenized.${l}-en.en >> $prep/train.src
    awk '{print "<'$l'> "$0}' $tmp/multiun.bpe.tokenized.${l}-en.${l} >> $prep/train.tgt
done

echo "train data done."

# # Tokenize and encode devset and testset
for l in ar ru en zh; do
    subword-nmt apply-bpe -c $BPE_CODE < $token_data/valid.$l > $tmp/valid.bpe.$l
    subword-nmt apply-bpe -c $BPE_CODE < $token_data/test.$l > $tmp/test.bpe.$l
done

# supervised
SRCS=(
    "ar-en"
    "en-ar"
    "zh-en"
    "en-zh"
    "ru-en"
    "en-ru"
)

# echo "pre-processing valid data..."
for SRC in "${SRCS[@]}"; do
    echo $SRC
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]} 
    echo $src,$tgt

    awk '{print "<'$src'> "$0}' $tmp/valid.bpe.$src | head -n 1000 >> $prep/valid.src
    awk '{print "<'$tgt'> "$0}' $tmp/valid.bpe.$tgt | head -n 1000 >> $prep/valid.tgt
done
# M

# zero-shot
SRCS=(
    "ru-zh"
    "ru-ar"
    "zh-ru"
    "zh-ar"
    "ar-zh"
    "ar-ru"
)

# echo "pre-processing valid data..."
for SRC in "${SRCS[@]}"; do
    echo $SRC
    array=(${SRC//-/ })
    src=${array[0]}
    tgt=${array[1]} 
    echo $src,$tgt

    awk '{print "<'$src'> "$0}' $tmp/valid.bpe.$src | head -n 1000 >> $prep/test.src
    awk '{print "<'$tgt'> "$0}' $tmp/valid.bpe.$tgt | head -n 1000 >> $prep/test.tgt
done
# M

fairseq-preprocess --source-lang src --target-lang tgt \
    --trainpref $prep/train \
    --validpref $prep/valid \
    --testpref $prep/test \
    --destdir $prep/data-bin \
    --workers 64 \
    --joined-dictionary 

signature=base_lcs

repo_dir=$( cd $( dirname $0 )/../.. && pwd )
data_dir=$repo_dir/data/ted/s_enc_t_dec
code_dir=$repo_dir/fairseq/fairseq-converter
model_dir=$repo_dir/checkpoints/ted/$signature

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=${repo_dir}:$PYTHONPATH

if [ ! -d ${model_dir} ]; then
    mkdir -p ${model_dir}
    touch ${model_dir}/train.log
    chmod -R 777 ${model_dir}
fi

python $code_dir/fairseq_cli/train.py $data_dir/data-bin \
    --save-dir $model_dir \
    --task translation_label \
    --arch transformer_wmt_en_de \
    --share-all-embeddings \
    --encoder-layers 6 --encoder-converter-layer 5 --decoder-layers 6 \
    --label-style 's-enc-t-dec' \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_le --label-smoothing 0.1 \
    --warmup-init-lr 1e-07 \
    --stop-min-lr -1 \
    --max-tokens 3200 \
    --max-update 100000 --save-interval-updates 2500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --scoring sacrebleu \
    --valid-subset valid,test \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend=no_c10d \
    --update-freq 2 \
    --num-workers 8 \
    --no-epoch-checkpoints \
    --seed 42 \
    --fp16 |&tee -a $model_dir/train.log

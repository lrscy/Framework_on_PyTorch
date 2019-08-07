DSTC8_HOME=./
cd ${DSTC8_HOME}
export EMBED_DIR=data/bert-embedding/
export DATA_DIR=data/MRPC/

lr=2e-5
ep=5
dp=0.5
b=32
s=128
wp=0.1
run=1

CUDA_VISIBLE_DEVICES=2,3 python src/main.py \
  --data_dir ${DATA_DIR} \
  --bert_dir ${EMBED_DIR} \
  --bert_file bert-base-cased \
  --do_train \
  --do_eval \
  --do_predict \
  --learning_rate ${lr} \
  --epoch ${ep} \
  --use_cuda \
  --batch_size ${b} \
  --max_seq_length ${s} \
  --warmup_propotion ${wp} \
  --dropout ${dp}\
  --output_dir results/dstc_output_lr${lr}_ep${ep}_dp${dp}_b${b}_s${s}_wp${wp}_run${run}/ \
  --shell_print shell \
  --suffix last \
  --multi_gpu

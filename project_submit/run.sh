# !/bin/sh

mkdir data

mkdir output_bert
mkdir output_nezha
mkdir pre_bert
mkdir pre_nezha
mkdir model

python build_corpus.py
python build_vocab.py

python run_training_bert.py &
python run_training_nezha.py &
wait

cp data/vocab.txt output_bert/checkpoint-703125/
cp data/vocab.txt output_nezha/checkpoint-351562/

# 1 epoch 测试
# cp data/vocab.txt output_bert/checkpoint-3515/
# cp data/vocab.txt output_nezha/checkpoint-3515/

# cp data/vocab.txt pre_bert/
# cp data/vocab.txt pre_nezha/

python train_v1_single.py &
python train_nezha_single.py &
wait

python onnx_convert_bert.py
python onnx_convert_nezha.py

python inference_onnx_merge.py

# docker build -t registry.cn-shanghai.aliyuncs.com/xiaoqifeng/gaiic_track3:3.0 .
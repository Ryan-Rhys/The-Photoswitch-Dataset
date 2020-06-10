#! /bin/bash

task='e_iso_pi'   # (e_iso_pi, z_iso_pi, e_iso_n, z_iso_n)
path='../dataset/photoswitches.csv'
representation='fingerprints' #(fingerprints, fragments, fragprints)


n_trials=50

r_size=8
det_encoder_hidden_size=32
det_encoder_n_hidden=2
lat_encoder_hidden_size=32
lat_encoder_n_hidden=2
decoder_hidden_size=32
decoder_n_hidden=2
batch_size=10
iterations=500
lr=0.001

mkdir -p $task
mkdir -p $task/results
mkdir -p $task/results/anp

python predict_with_ANP.py --task $task --path $path \
  --representation ${representation} --n_trials ${n_trials} \
  --r_size ${r_size} --det_encoder_hidden_size ${det_encoder_hidden_size} \
  --det_encoder_n_hidden ${det_encoder_n_hidden} \
  --lat_encoder_hidden_size ${lat_encoder_hidden_size} \
  --lat_encoder_n_hidden ${lat_encoder_n_hidden} \
  --decoder_hidden_size ${decoder_hidden_size} \
  --decoder_n_hidden ${decoder_n_hidden} \
  --batch_size ${batch_size} --iterations ${iterations} \
  --lr ${lr}

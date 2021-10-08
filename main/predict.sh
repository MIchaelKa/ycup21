PYTHONPATH=./py_packages:$PYTHONPATH python3 predict.py \
  --ckpt_path checkpoints/epoch=0-step=3-inference.ckpt \
  --data_directory $1 --predicts_file $2

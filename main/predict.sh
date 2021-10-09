PYTHONPATH=./py_packages:$PYTHONPATH python3 predict.py \
  --ckpt_path checkpoints/epoch=17-step=12455-inference.ckpt \
  --data_directory $1 --predicts_file $2

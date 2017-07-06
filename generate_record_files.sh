PATH2MXNET=$HOME'/packages/mxnet'

python3.4 $PATH2MXNET/tools/im2rec.py ./records/test_fold_1 ./records/
python3.4 $PATH2MXNET/tools/im2rec.py ./records/train_fold_1 ./records/


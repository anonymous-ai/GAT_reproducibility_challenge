DATASET=$1
NUM_TR_PER_CLASS=$2
LAYER_TYPE=$3

cmd="python run.py  \
    --dataset $DATASET \
    --num_train_per_class $NUM_TR_PER_CLASS \
    --layer_type $LAYER_TYPE"

echo $cmd
eval $cmd
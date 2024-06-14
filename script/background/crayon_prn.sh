cd ../..
CURR_PATH=$(pwd)
cd ${CURR_PATH}

CUDA_NO=0

DATA_DIR="${CURR_PATH}/data/background"
EXPL_SAVE_DIR="${CURR_PATH}/interpretation"
HFB_SAVE_DIR="${CURR_PATH}/annotation"
MODEL_SAVE_DIR="${CURR_PATH}/model"
PERFORMANCE_SAVE_DIR="${CURR_PATH}/performance"

PERFORMANCE_SAVE_NAME="background_crayon_prn.csv"
MODEL_SAVE_NAME="background_resnet50.pt"
NUM_FB=-1

cd src
python3 background.py \
    --data_name background \
    --model_name resnet50 \
    --method crayon-pruning \
    --num_workers 4 \
    --batch_size 256 \
    --w_cls 1 \
    --fb_epoch 10 \
    --fb_lr 0.000005 \
    --weight_decay 0.1 \
    --num_fb ${NUM_FB} \
    --cuda ${CUDA_NO} \
    --model_dir ${MODEL_SAVE_DIR} \
    --model_save_name ${MODEL_SAVE_NAME} \
    --data_dir ${DATA_DIR} \
    --expl_dir ${EXPL_SAVE_DIR} \
    --hfb_dir ${HFB_SAVE_DIR} \
    --performance_save_on \
    --performance_save_dir ${PERFORMANCE_SAVE_DIR} \
    --performance_save_name ${PERFORMANCE_SAVE_NAME}
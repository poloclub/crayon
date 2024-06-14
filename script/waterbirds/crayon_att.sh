cd ../..
CURR_PATH=$(pwd)
cd ${CURR_PATH}

CUDA_NO=0

DATA_DIR="${CURR_PATH}/data/waterbirds"
EXPL_SAVE_DIR="${CURR_PATH}/interpretation"
HFB_SAVE_DIR="${CURR_PATH}/annotation"
MODEL_SAVE_DIR="${CURR_PATH}/model"
PERFORMANCE_SAVE_DIR="${CURR_PATH}/performance"

PERFORMANCE_SAVE_NAME="waterbirds_crayon_att.csv"
MODEL_SAVE_NAME="waterbirds_resnet50.pth"
NUM_FB=-1

cd src
python3 waterbirds.py \
    --data_name waterbirds \
    --model_name resnet50 \
    --method crayon-attention \
    --num_workers 4 \
    --batch_size 128 \
    --w_yes 1e7 \
    --w_no 2e5 \
    --w_cls 1 \
    --weight_decay 0.0001 \
    --fb_epoch 10 \
    --fb_lr 0.00005 \
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

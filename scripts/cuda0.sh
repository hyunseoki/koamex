DEVICE = "cuda:0"

###############################################################################################
MODEL = "smp"
COMMENTS = "cos_scheduler_T25_w_L1Loss"
BASE_PATH = "./data/resized_scale5"
SAVE_FOLDER = "./checkpoint/$COMMENTS"

python main.py --epochs 600 \
                --batch_size 16 \
                --lr 1e-3 \
                --T0 25 \
                --device $DEVICE \
                --model $MODEL \
                --num_kp 36 \
                --base_path $BASE_PATH \
                --save_folder $SAVE_FOLDER \
                --comments $COMMENTS


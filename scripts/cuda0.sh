$DEVICE = "cuda:0"

###############################################################################################
# $model = "smp"
# $comments = "cos_scheduler_T25_w_L1Loss"
# $base_path = "./data/resized_scale5"
# $save_folder = "./checkpoint/$comments"

python main.py
            # --epochs 600 \
            # --batch_size 16 \
            # --lr 1e-3 \
            # --T0 25 \
            --device $DEVICE \
            # --model $model \
            # --num_kp 36 \
            # --base_path $base_path \
            # --save_folder $save_folder \
            # --comments $comments


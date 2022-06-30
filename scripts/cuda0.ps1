function PyScript{
    param([String]$cmd)
    $host.UI.RawUI.WindowTitle = $cmd
    Invoke-Expression -Command $cmd
}

## activate venv
C:\src\py\venv\hyunseoki\Scripts\Activate.ps1

$device = "cuda:0"

###############################################################################################
$model = "smp"
$comments = "cos_scheduler_T25_w_L1Loss"
$base_path = "./data/resized_scale5"
$save_folder = "./checkpoint/$comments"

$train = "python main.py"  +
            " --epochs 600 " +
            " --batch_size 16 " +
            " --lr 1e-3 " +
            " --T0 25 " +
            " --device $device " +
            " --model $model" +
            " --num_kp 36" + 
            " --base_path $base_path " +
            " --save_folder $save_folder" +
            " --comments $comments"

PyScript($train)

# ################################################################################################
# $model = "smp"
# $comments = "kp36_cos_scheduler_T25"
# $base_path = "./data/resized_scale5"
# $save_folder = "./checkpoint/$comments"

# $train = "python main.py"  +
#             " --epochs 150 " +
#             " --batch_size 16 " +
#             " --lr 1e-3 " +
#             " --T0 25 " +
#             " --device $device " +
#             " --model $model" +
#             " --num_kp 36" + 
#             " --base_path $base_path " +
#             " --save_folder $save_folder" +
#             " --comments $comments"

# PyScript($train)

# ################################################################################################
# $model = "smp"
# $comments = "kp36_cos_scheduler_T50"
# $base_path = "./data/resized_scale5"
# $save_folder = "./checkpoint/$comments"

# $train = "python main.py"  +
#             " --epochs 150 " +
#             " --batch_size 16 " +
#             " --lr 1e-3 " +
#             " --T0 50 " +
#             " --device $device " +
#             " --model $model" +
#             " --num_kp 36" + 
#             " --base_path $base_path " +
#             " --save_folder $save_folder" +
#             " --comments $comments"

# PyScript($train)
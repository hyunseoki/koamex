function PyScript{
    param([String]$cmd)
    $host.UI.RawUI.WindowTitle = $cmd
    Invoke-Expression -Command $cmd
}

$device = "cuda:1"
################################################################################################
$model = "smp"
$comments = "scale5_w_mre_kp30"
$base_path = "./data/resized_scale5"
$save_folder = "./checkpoint/$comments"

$train = "python main.py"  +
            " --epochs 100 " +
            " --batch_size 16 " +
            " --lr 1e-3 " +
            " --device $device " +
            " --model $model" +
            " --num_kp 30" + 
            " --base_path $base_path " +
            " --save_folder $save_folder" +
            " --use_mre_as_loss True" + 
            " --comments $comments"

PyScript($train)

################################################################################################
$model = "smp"
$comments = "scale5_wo_mre_kp30"
$base_path = "./data/resized_scale5"
$save_folder = "./checkpoint/$comments"

$train = "python main.py"  +
            " --epochs 100 " +
            " --batch_size 16 " +
            " --lr 1e-3 " +
            " --device $device " +
            " --model $model" +
            " --num_kp 30" + 
            " --base_path $base_path " +
            " --save_folder $save_folder" +
            " --use_mre_as_loss False" + 
            " --comments $comments"

PyScript($train)
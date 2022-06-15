function PyScript{
    param([String]$cmd)
    $host.UI.RawUI.WindowTitle = $cmd
    Invoke-Expression -Command $cmd
}

$device = "cuda:0"
################################################################################################
$model = "smp"
$comments = "scale5_wo_mre_"
$base_path = "./data/resized_scale5"
$save_folder = "./checkpoint/$comments"

$train = "python main.py"  +
            " --epochs 50 " +
            " --batch_size 16 " +
            " --lr 1e-3 " +
            " --device $device " +
            " --model $model" +
            " --base_path $base_path " +
            " --save_folder $save_folder" +
            " --use_mre_as_loss false" + 
            " --comments $comments"

PyScript($train)
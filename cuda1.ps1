function PyScript{
    param([String]$cmd)
    $host.UI.RawUI.WindowTitle = $cmd
    Invoke-Expression -Command $cmd
}

python src/gen_heatmap.py

$device = "cuda:1"
################################################################################################
$model = "smp"
$comments = "scale5_w_mre"
$base_path = "./data/resized_scale5"
$save_folder = "./checkpoint/$comments"

$train = "python main.py"  +
            " --epochs 100 " +
            " --batch_size 16 " +
            " --lr 1e-3 " +
            " --device $device " +
            " --model $model" +
            " --base_path $base_path " +
            " --save_folder $save_folder" +
            " --use_mre_as_loss true" + 
            " --comments $comments"

PyScript($train)

################################################################################################
$model = "smp"
$comments = "scale5_wo_mre"
$base_path = "./data/resized_scale5"
$save_folder = "./checkpoint/$comments"

$train = "python main.py"  +
            " --epochs 100 " +
            " --batch_size 16 " +
            " --lr 1e-3 " +
            " --device $device " +
            " --model $model" +
            " --base_path $base_path " +
            " --save_folder $save_folder" +
            " --use_mre_as_loss false" + 
            " --comments $comments"

PyScript($train)
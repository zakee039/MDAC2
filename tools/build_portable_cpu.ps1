param([string]$Output = (Join-Path (Split-Path -Parent $PSScriptRoot) 'MDAC2-v0.4.0-portable-cpu.zip'), [switch]$KeepStage)
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Output = [IO.Path]::GetFullPath($Output)
if(Test-Path -LiteralPath $Output){throw "Refusing to overwrite existing package: $Output"}
$stage = Join-Path ([IO.Path]::GetTempPath()) ('mdac2-portable-' + [guid]::NewGuid().ToString('N'))
$root = Join-Path $stage 'MDAC2-v0.4.0-portable-cpu'
New-Item -ItemType Directory -Force $root | Out-Null

function AssertStagePath([string]$path) {
  $resolved = [IO.Path]::GetFullPath($path)
  $base = [IO.Path]::GetFullPath($stage)
  if($resolved -ne $base -and -not $resolved.StartsWith($base + [IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw "Path outside build stage: $resolved"}
  if(Test-Path -LiteralPath $resolved){
    $items = @((Get-Item -LiteralPath $resolved -Force)) + @(Get-ChildItem -LiteralPath $resolved -Recurse -Force)
    if($items | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }){throw "Reparse point in build stage: $resolved"}
  }
}

function CopyPath([string]$relative) {
  $from=Join-Path $source $relative; $to=Join-Path $root $relative
  if(-not(Test-Path -LiteralPath $from)){throw "Missing required path: $relative"}
  New-Item -ItemType Directory -Force (Split-Path -Parent $to) | Out-Null
  Copy-Item -LiteralPath $from -Destination $to -Recurse -Force
}
function Robo([string]$from,[string]$to,[string[]]$extra=@()) {
  New-Item -ItemType Directory -Force $to | Out-Null
  & robocopy $from $to /E /R:1 /W:1 /NFL /NDL /NJH /NJS @extra | Out-Null
  if($LASTEXITCODE -gt 7){throw "robocopy failed ($LASTEXITCODE): $from"}
}

# The studio exposes R2/V2/V3/V4; V4 reuses V3 weights. Keep V1 files in
# the repository, but do not make every portable user carry them.
Robo (Join-Path $source 'src') (Join-Path $root 'src') @('/XD','__pycache__','*.egg-info')
CopyPath 'models/v2'; CopyPath 'models/v3'; CopyPath 'models/v4'; CopyPath 'models/registry.json'; CopyPath 'recipes'; CopyPath 'web/dist'; CopyPath 'start.bat'; CopyPath 'README.md'; CopyPath 'CHANGELOG.md'
CopyPath 'tools/studio_generate.py'; CopyPath 'tools/generate_joint_playtest.py'; CopyPath 'tools/studio_detect_timing.py'; CopyPath 'tools/studio_detect_trim.py'
CopyPath 'tools/private/beat-this'; CopyPath 'tools/experiments/beat_this/models.json'

# Only the pinned RoFormer inputs are needed at runtime. The comparison and
# smoke stores are development artifacts and contain tens of megabytes of audio.
foreach($relative in @(
  'tools/private/separation/MelBandRoformer.ckpt',
  'tools/private/separation/roformer-author-config.yaml',
  'tools/private/separation/roformer-manifest.json',
  'tools/private/separation/roformer-config-source.json',
  'tools/private/separation/source-LICENSE'
)){ CopyPath $relative }

# FFmpeg is used through ffmpeg.exe and ffprobe.exe only. Preserve the license
# and expected directory shape, while omitting the archive, docs and ffplay.
$ffmpegSource=Join-Path $source 'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build'
$ffmpegBin=Join-Path $root 'tools/.deps/ffmpeg/extracted/ffmpeg-9.0.1-essentials_build/bin'
New-Item -ItemType Directory -Force $ffmpegBin | Out-Null
Copy-Item -LiteralPath (Join-Path $ffmpegSource 'bin/ffmpeg.exe') -Destination $ffmpegBin -Force
Copy-Item -LiteralPath (Join-Path $ffmpegSource 'bin/ffprobe.exe') -Destination $ffmpegBin -Force
Copy-Item -LiteralPath (Join-Path $ffmpegSource 'LICENSE') -Destination (Join-Path (Split-Path $ffmpegBin -Parent) 'LICENSE') -Force

$apiBase='C:\Users\mamin\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none'
if(-not(Test-Path $apiBase)){throw "Missing Python 3.12 runtime: $apiBase"}
$apiRoot=Join-Path $root 'runtime/python'; Robo $apiBase $apiRoot @('/XD','__pycache__','*site-packages*')
Robo (Join-Path $source '.venv/Lib/site-packages') (Join-Path $apiRoot 'Lib/site-packages') @('/XD','__pycache__')
# Replace checkout-bound editable imports with a relocatable source reference.
Get-ChildItem -LiteralPath (Join-Path $apiRoot 'Lib/site-packages') -File -Filter '__editable__*mdac2*' | Remove-Item -Force
'../../../../src' | Set-Content (Join-Path $apiRoot 'Lib/site-packages/mdac2-portable.pth') -Encoding ASCII
Get-ChildItem -LiteralPath (Join-Path $apiRoot 'Lib/site-packages') -Directory -Filter 'mdac2-*.dist-info' | ForEach-Object {
  $direct = Join-Path $_.FullName 'direct_url.json'
  if(Test-Path -LiteralPath $direct){Remove-Item -LiteralPath $direct -Force}
}
foreach($optional in @('numba','llvmlite','librosa','torch','torchaudio','torchvision')) {
  $candidate=Join-Path $apiRoot ("Lib/site-packages/"+$optional)
  if(Test-Path $candidate){AssertStagePath $candidate; Remove-Item -LiteralPath $candidate -Recurse -Force}
}

$workerBase='D:\anaconda'; if(-not(Test-Path $workerBase)){throw "Missing Python 3.13 runtime: $workerBase"}
$workerRoot=Join-Path $root 'runtime/python313'; Robo $workerBase $workerRoot @('/XD','__pycache__','pkgs','conda-bld','Lib/site-packages')
Robo (Join-Path $source 'tools/.deps/roformer-py313/Lib/site-packages') (Join-Path $workerRoot 'Lib/site-packages') @('/XD','__pycache__')
Robo (Join-Path $source 'tools/.deps/beat-this-py313/Lib/site-packages') (Join-Path $workerRoot 'Lib/site-packages') @('/XD','__pycache__')

# The copied environments are execution-only. Static import libraries, debug
# symbols, bytecode caches and conda's package manager are not needed to run.
foreach($runtimeRoot in @($apiRoot,$workerRoot)) {
  AssertStagePath $runtimeRoot
  Get-ChildItem -LiteralPath $runtimeRoot -Recurse -Force -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @('.pdb','.lib','.pyc','.pyo') } |
    Remove-Item -Force -ErrorAction SilentlyContinue
}
foreach($relative in @('Library/lib','Library/include','include','libs','Scripts','condabin','conda-meta','share')) {
  $candidate=Join-Path $workerRoot $relative
  if(Test-Path -LiteralPath $candidate){AssertStagePath $candidate; Remove-Item -LiteralPath $candidate -Recurse -Force}
}
foreach($file in @('_conda.exe','Uninstall-Miniconda3.exe','.condarc','EULA.txt','LICENSE_PYTHON.txt','install.log','pre_uninstall.bat')) {
  $candidate=Join-Path $workerRoot $file
  if(Test-Path -LiteralPath $candidate){[IO.File]::Delete($candidate)}
}

@{application_version='0.4.0';runtime='cpu';api_python='3.12';worker_python='3.13';entrypoint='start.bat';default_strategy='v4';strategies=@('R2','V2','V3','V4');weight_models=@('V2','V3');excludes=@('CUDA runtime','GPU environment','V1 historical weights','development audio stores','FFmpeg docs/archive','debug symbols/static import libraries','private project data','chart runtime database')} |
  ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'PORTABLE-MANIFEST.json') -Encoding UTF8
& 7z.exe a -tzip -mx=1 -mmt=on -y $Output $root | Out-Null
if($LASTEXITCODE -ne 0){throw "7-Zip failed ($LASTEXITCODE)"}
Write-Output "Created $Output"
if($KeepStage){Write-Output "Stage retained for verification: $root"}
else {AssertStagePath $stage; Remove-Item -LiteralPath $stage -Recurse -Force}

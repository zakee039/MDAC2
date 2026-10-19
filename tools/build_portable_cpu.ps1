param([string]$Output = (Join-Path (Split-Path -Parent $PSScriptRoot) 'MDAC2-v0.3.0-portable-cpu.zip'))
$ErrorActionPreference = 'Stop'
$source = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stage = Join-Path ([IO.Path]::GetTempPath()) ('mdac2-portable-' + [guid]::NewGuid().ToString('N'))
$root = Join-Path $stage 'MDAC2-v0.3.0-portable-cpu'
New-Item -ItemType Directory -Force $root | Out-Null

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

CopyPath 'src'; CopyPath 'models'; CopyPath 'recipes'; CopyPath 'web/dist'; CopyPath 'start.bat'; CopyPath 'README.md'; CopyPath 'CHANGELOG.md'
CopyPath 'tools/studio_generate.py'; CopyPath 'tools/generate_joint_playtest.py'; CopyPath 'tools/studio_detect_timing.py'; CopyPath 'tools/studio_detect_trim.py'
CopyPath 'tools/private/beat-this'; CopyPath 'tools/private/separation'; CopyPath 'tools/.deps/ffmpeg'; CopyPath 'tools/experiments/beat_this/models.json'

$apiBase='C:\Users\mamin\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none'
if(-not(Test-Path $apiBase)){throw "Missing Python 3.12 runtime: $apiBase"}
$apiRoot=Join-Path $root 'runtime/python'; Robo $apiBase $apiRoot @('/XD','__pycache__','*site-packages*')
Robo (Join-Path $source '.venv/Lib/site-packages') (Join-Path $apiRoot 'Lib/site-packages') @('/XD','__pycache__')

$workerBase='D:\anaconda'; if(-not(Test-Path $workerBase)){throw "Missing Python 3.13 runtime: $workerBase"}
$workerRoot=Join-Path $root 'runtime/python313'; Robo $workerBase $workerRoot @('/XD','__pycache__','pkgs','conda-bld','Lib/site-packages')
Robo (Join-Path $source 'tools/.deps/roformer-py313/Lib/site-packages') (Join-Path $workerRoot 'Lib/site-packages') @('/XD','__pycache__')
Robo (Join-Path $source 'tools/.deps/beat-this-py313/Lib/site-packages') (Join-Path $workerRoot 'Lib/site-packages') @('/XD','__pycache__')

@{application_version='0.3.0';runtime='cpu';api_python='3.12';worker_python='3.13';entrypoint='start.bat';models=@('V2','V3','R2 adapter');excludes=@('CUDA runtime','GPU environment','private project data','chart runtime database')} |
  ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'PORTABLE-MANIFEST.json') -Encoding UTF8
if(Test-Path -LiteralPath $Output){[IO.File]::Delete((Resolve-Path $Output).Path)}
& 7z.exe a -tzip -mx=1 -mmt=on -y $Output $root | Out-Null
if($LASTEXITCODE -ne 0){throw "7-Zip failed ($LASTEXITCODE)"}
Write-Output "Created $Output"
[IO.Directory]::Delete($stage,$true)

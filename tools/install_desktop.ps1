# Build a WangYun desktop shortcut (double-click to open the local app).
# Run:  powershell -ExecutionPolicy Bypass -File tools\install_desktop.ps1
$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
Write-Host "[1/4] project: $root"

# --- find a Python to bootstrap the venv -------------------------------------
$py = $env:WY_PYTHON
if (-not $py) {
    foreach ($cand in @('python', 'py')) {
        $cmd = Get-Command $cand -ErrorAction SilentlyContinue
        # 跳过 Windows Store 的 python.exe 占位符（它不会真的建 venv）
        if ($cmd -and $cmd.Source -notmatch '\\WindowsApps\\') { $py = $cmd.Source; break }
    }
}
if (-not $py -or -not (Test-Path $py)) {
    $fallback = Join-Path $env:USERPROFILE '.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
    if (Test-Path $fallback) { $py = $fallback }
}
if (-not $py) { throw 'Python 3.10+ not found. Install it or set WY_PYTHON.' }
Write-Host "[2/4] python : $py"

# --- project-local venv (self-contained; keeps running if DSH updates) -------
$venv = Join-Path $root '.venv'
$vpy  = Join-Path $venv 'Scripts\python.exe'
$vpw  = Join-Path $venv 'Scripts\pythonw.exe'
if (-not (Test-Path $vpy)) {
    Write-Host '[3/4] creating .venv ...'
    & $py -m venv $venv
}
Write-Host '[3/4] installing wangyun (editable) ...'
& $vpy -m pip install --quiet --disable-pip-version-check --upgrade pip
& $vpy -m pip install --quiet --disable-pip-version-check -e $root
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }

# --- icon --------------------------------------------------------------------
$ico = Join-Path $root 'assets\wangyun.ico'
if (-not (Test-Path $ico)) {
    & $vpy (Join-Path $root 'tools\make_icon.py')
}

# --- shortcuts ---------------------------------------------------------------
function New-Shortcut([string]$path) {
    $ws = New-Object -ComObject WScript.Shell
    $sc = $ws.CreateShortcut($path)
    $sc.TargetPath = $vpw
    $sc.Arguments = '-m wy.desktop'
    $sc.WorkingDirectory = $root
    $sc.IconLocation = "$ico,0"
    $sc.Description = 'WangYun- observer of shapes grown from relations'
    $sc.Save()
}

$desktop = [Environment]::GetFolderPath('Desktop')
$startMenu = [Environment]::GetFolderPath('Programs')

New-Shortcut (Join-Path $desktop 'WangYun.lnk')
New-Shortcut (Join-Path $startMenu 'WangYun.lnk')
Write-Host "[4/4] shortcuts: $desktop\WangYun.lnk  (and Start Menu)"

# --- smoke: import + version -------------------------------------------------
& $vpy -c "import wy, wy.desktop; print('ok', wy.__version__)"
Write-Host 'Done. Double-click the WangYun icon on your desktop.'

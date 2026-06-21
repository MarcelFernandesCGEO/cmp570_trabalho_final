# run_pipeline.ps1 - Pipeline completo: preparar chips -> treinar -> inferir -> avaliar
#
# Uso basico (usa os dados e configs padrao):
#   .\run_pipeline.ps1
#
# Uso com novos dados:
#   .\run_pipeline.ps1 -PlanetDir C:\dados\planet -GtTif C:\dados\gt.tif -OutData C:\dados\chips
#
# Parametros:
#   -PlanetDir   Pasta com TIFs Planet (default: ..\data\planet)
#   -GtTif       TIF ground truth aerofoto (default: ..\data\gt\gt.tif)
#   -OutData     Onde salvar os chips (default: C:\Users\marcel.1CGEO\Desktop\zero_shot_data)
#   -ConfigsDir  Pasta com YAMLs de treino (default: ..\configs)
#   -ResultsDir  Pasta de resultados (default: ..\results)
#   -SkipPrepare Pular prepare_chips se chips ja existem
#   -SkipTrain   Pular treino (so infer + evaluate)
#   -SkipInfer   Pular inferencia (so evaluate)
#   -Checkpoint  Qual checkpoint usar: "best" ou numero (default: 5000)

param(
    [string]$PlanetDir  = "$PSScriptRoot\..\data\planet",
    [string]$GtTif      = "$PSScriptRoot\..\data\gt\gt.tif",
    [string]$OutData    = "C:\Users\marcel.1CGEO\Desktop\zero_shot_data",
    [string]$ConfigsDir = "$PSScriptRoot\..\configs",
    [string]$ResultsDir = "$PSScriptRoot\..\results",
    [switch]$SkipPrepare,
    [switch]$SkipTrain,
    [switch]$SkipInfer,
    [string]$Checkpoint = "5000"
)

$ErrorActionPreference = "Stop"

# Caminhos fixos
$SATLAS   = "C:\Users\marcel.1CGEO\Desktop\GitHub\satlas-super-resolution"
$SCRIPTS  = $PSScriptRoot
$EVALSCRIPT = "$PSScriptRoot\..\..\evaluate_ffl.py"

$env:PYTHONPATH      = $SATLAS
$env:PYTHONUNBUFFERED = "1"

Write-Host ""
Write-Host "========================================"
Write-Host " Pipeline SATLAS + Focal Frequency Loss "
Write-Host "========================================"
Write-Host ""

# ------------------------------------------------------------------
# 1. Preparar chips
# ------------------------------------------------------------------
if (-not $SkipPrepare) {
    Write-Host "[1/4] Preparando chips..."
    python "$SCRIPTS\prepare_chips.py" `
        --planet_dir $PlanetDir `
        --gt_tif     $GtTif `
        --out_dir    $OutData
    if ($LASTEXITCODE -ne 0) { throw "prepare_chips.py falhou" }
    Write-Host ""
} else {
    Write-Host "[1/4] Preparar chips: PULADO"
}

# ------------------------------------------------------------------
# 2. Treinar (3 variantes)
# ------------------------------------------------------------------
if (-not $SkipTrain) {
    Write-Host "[2/4] Treinando variantes..."
    $configs = @(
        @{ yml = "$ConfigsDir\baseline_from_pretrained.yml";  name = "baseline" },
        @{ yml = "$ConfigsDir\ffl_w01_from_pretrained.yml";   name = "ffl_w01"  },
        @{ yml = "$ConfigsDir\ffl_w10_from_pretrained.yml";   name = "ffl_w10"  }
    )
    foreach ($c in $configs) {
        Write-Host "  Treinando $($c.name)..."
        Push-Location $SATLAS
        python -m ssr.train -opt $c.yml
        Pop-Location
        if ($LASTEXITCODE -ne 0) { throw "Treino $($c.name) falhou" }
    }
    Write-Host ""
} else {
    Write-Host "[2/4] Treino: PULADO"
}

# ------------------------------------------------------------------
# 3. Inferencia
# ------------------------------------------------------------------
if (-not $SkipInfer) {
    Write-Host "[3/4] Rodando inferencia..."

    $variants = @(
        @{ name = "baseline"; exp = "baseline_from_pretrained" },
        @{ name = "ffl_w01";  exp = "ffl_w01_from_pretrained"  },
        @{ name = "ffl_w10";  exp = "ffl_w10_from_pretrained"  }
    )

    foreach ($v in $variants) {
        $ckpt = "$ResultsDir\$($v.name)\$($v.exp)\models\net_g_$Checkpoint.pth"
        if (-not (Test-Path $ckpt)) {
            Write-Host "  [AVISO] Checkpoint nao encontrado: $ckpt"
            continue
        }
        $srOut = "$ResultsDir\$($v.name)\sr_flat"
        Write-Host "  $($v.name) -> $srOut"
        python "$SCRIPTS\infer_chips.py" `
            --checkpoint $ckpt `
            --planet_dir "$OutData\planet_chips" `
            --out_dir    $srOut `
            --gt_dir     "$OutData\gt_chips"
        if ($LASTEXITCODE -ne 0) { throw "Inferencia $($v.name) falhou" }
    }
    Write-Host ""
} else {
    Write-Host "[3/4] Inferencia: PULADO"
}

# ------------------------------------------------------------------
# 4. Avaliacao
# ------------------------------------------------------------------
Write-Host "[4/4] Avaliando..."

$srDirs = @(
    "$ResultsDir\baseline\sr_flat",
    "$ResultsDir\ffl_w01\sr_flat",
    "$ResultsDir\ffl_w10\sr_flat"
)
$gtFlat = "$ResultsDir\baseline\gt_flat"

# Verificar se os diretórios SR existem
$existingSrDirs = @()
$existingNames  = @()
$nameMap = @("baseline", "ffl_w01", "ffl_w10")
for ($i = 0; $i -lt $srDirs.Length; $i++) {
    if (Test-Path $srDirs[$i]) {
        $existingSrDirs += $srDirs[$i]
        $existingNames  += $nameMap[$i]
    }
}

if ($existingSrDirs.Length -eq 0) {
    Write-Host "[AVISO] Nenhum diretorio SR encontrado. Rode sem -SkipInfer primeiro."
} elseif (-not (Test-Path $gtFlat)) {
    Write-Host "[AVISO] gt_flat nao encontrado em $gtFlat"
    Write-Host "        Rode infer_chips.py com --gt_dir para gera-lo."
} else {
    $evalOut = "$ResultsDir\evaluation"
    python $EVALSCRIPT `
        --sr_dirs $existingSrDirs `
        --gt_dir  $gtFlat `
        --names   $existingNames `
        --out_dir $evalOut
    if ($LASTEXITCODE -ne 0) { throw "evaluate_ffl.py falhou" }
    Write-Host ""
    Write-Host "Resultados em: $evalOut"
}

Write-Host ""
Write-Host "========================================"
Write-Host " Pipeline concluido!"
Write-Host "========================================"

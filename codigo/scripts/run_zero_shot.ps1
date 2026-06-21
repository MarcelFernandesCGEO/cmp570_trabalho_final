# run_zero_shot.ps1 — Ablation study do Projeto Final CMP570 (Windows)
#
# Roda as 3 variantes sequencialmente partindo dos pesos SATLAS pré-treinados.
# Antes de rodar: ativar o venv e garantir que o SATLAS está clonado.
#
# Uso:
#   cd C:\Users\marcel.1CGEO\Desktop\GitHub\satlas-super-resolution
#   .\.venv\Scripts\Activate.ps1
#   .\caminho\para\run_zero_shot.ps1

$ErrorActionPreference = "Stop"

$SATLAS   = "C:\Users\marcel.1CGEO\Desktop\GitHub\satlas-super-resolution"
$CONFIGS  = "C:\Users\marcel.1CGEO\Desktop\GitHub\cmp570\trabalhos\trabalho_final\zero_shot\configs"

Set-Location $SATLAS

Write-Host "=== Verificando GPU ===" -ForegroundColor Cyan
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host ""
Write-Host "=== Variante 1/3: Baseline (sem FFL) ===" -ForegroundColor Green
python ssr/train.py --opt "$CONFIGS\baseline_from_pretrained.yml"

Write-Host ""
Write-Host "=== Variante 2/3: FFL λ=0.1 ===" -ForegroundColor Green
python ssr/train.py --opt "$CONFIGS\ffl_w01_from_pretrained.yml"

Write-Host ""
Write-Host "=== Variante 3/3: FFL λ=1.0 ===" -ForegroundColor Green
python ssr/train.py --opt "$CONFIGS\ffl_w10_from_pretrained.yml"

Write-Host ""
Write-Host "=== Ablation concluído ===" -ForegroundColor Cyan
Write-Host "Rode evaluate_ffl.py para gerar tabela e figuras PSD."

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    function Invoke-Checked {
        param([string]$Program, [string[]]$Arguments)
        & $Program @Arguments
        if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)." }
    }

    function Get-VerifiedFile($Url, $Destination, $Sha256) {
        if (Test-Path -LiteralPath $Destination) {
            if ((Get-FileHash -LiteralPath $Destination).Hash -eq $Sha256) { return }
        }
        $partial = "$Destination.partial"
        Invoke-Checked curl.exe @('--fail', '--location', '--retry', '2', '--continue-at', '-', '--output', $partial, $Url)
        if ((Get-FileHash -LiteralPath $partial).Hash -ne $Sha256) {
            throw "Checksum mismatch for $partial. Delete that partial file and retry."
        }
        Move-Item -LiteralPath $partial -Destination $Destination -Force
    }

    $revision = '7cb804a8609e9f4b1a50b7f31436d2df40bb9481'
    $patch = Join-Path $PSScriptRoot 'tag-scores.patch'
    if (!(Test-Path 'upstream/.git')) {
        Invoke-Checked git @('clone', '--no-checkout', 'https://github.com/xinyu1205/recognize-anything.git', 'upstream')
    }
    Invoke-Checked git @('-C', 'upstream', 'checkout', '--detach', $revision)
    & git -C upstream apply --reverse --check $patch 2>$null
    if ($LASTEXITCODE -ne 0) {
        Invoke-Checked git @('-C', 'upstream', 'apply', $patch)
    }

    if (!(Test-Path '.venv/Scripts/python.exe')) {
        Invoke-Checked py @('-3.11', '-m', 'venv', '.venv')
    }
    $python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    Invoke-Checked $python @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cpu')
    Invoke-Checked $python @('-m', 'pip', 'install', 'transformers==4.28.1', 'fairscale==0.4.13', 'timm==0.4.12', 'scipy', 'Pillow')

    New-Item -ItemType Directory -Path 'pretrained' -Force | Out-Null
    Get-VerifiedFile 'https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth' 'pretrained/ram_plus_swin_large_14m.pth' '497C178836BA66698CA226C7895317E6E800034BE986452DBD2593298D50E87D'
    Get-VerifiedFile 'https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt' 'pretrained/vocab.txt' '07ECED375CEC144D27C900241F3E339478DEC958F92FDDBC551F295C992038A3'
    Write-Output 'RAM++ setup complete. Run .venv\Scripts\python.exe classify.py "C:\path\to\photo.jpg".'
} finally {
    Pop-Location
}

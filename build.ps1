# Builds LiveTranscriber.exe into .\release and zips it for sharing.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File build.ps1
# Both the release folder and the zip are gitignored - they are regenerable.
Set-Location $PSScriptRoot
$release = Join-Path $PSScriptRoot "release"

Write-Host "Building with PyInstaller..." -ForegroundColor Cyan
# PyInstaller logs to stderr; route through cmd so PowerShell doesn't
# mistake INFO lines for errors. Success is judged by the exit code only.
cmd /c "python -m PyInstaller LiveTranscriber.spec --noconfirm --distpath ""$release"" --workpath build 2>&1"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed (exit $LASTEXITCODE)" }

# Intermediates are regenerable - don't keep them around.
Remove-Item build -Recurse -Force -ErrorAction SilentlyContinue

# Include the models.json template and README so friends get them too.
Copy-Item models.example.json "$release\LiveTranscriber\" -Force -ErrorAction SilentlyContinue
Copy-Item README.md "$release\LiveTranscriber\" -Force -ErrorAction SilentlyContinue

Write-Host "Zipping..." -ForegroundColor Cyan
$zip = "$release\LiveTranscriber-win64.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path "$release\LiveTranscriber" -DestinationPath $zip

$size = [math]::Round((Get-Item $zip).Length / 1MB, 1)
Write-Host "Done: $zip ($size MB)" -ForegroundColor Green
Write-Host "Send the zip to friends - they unzip anywhere writable and run LiveTranscriber.exe."
Write-Host "The zip is too big for git - attach it to a GitHub Release instead." -ForegroundColor DarkGray

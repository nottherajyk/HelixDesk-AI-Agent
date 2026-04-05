try {
    Write-Host "Running pip install..."
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Pip install failed" }

    Write-Host "Running pytest..."
    pytest tests/ -v
    if ($LASTEXITCODE -ne 0) { throw "Pytest failed" }

    Write-Host "Running baseline..."
    python baseline.py | Out-File -FilePath _final3.txt -Encoding ascii
    if ($LASTEXITCODE -ne 0) { throw "Baseline failed" }

    Write-Host "Running random agent train..."
    python train.py --agent random --episodes 10
    if ($LASTEXITCODE -ne 0) { throw "Random train failed" }

    Write-Host "Running rule agent train..."
    python train.py --agent rule --episodes 10
    if ($LASTEXITCODE -ne 0) { throw "Rule train failed" }

    Write-Host "Running bounds check..."
    python _verify.py | Out-File -FilePath _verify_out.txt -Encoding ascii
    if ($LASTEXITCODE -ne 0) { throw "Verify failed" }

    Write-Host "Git push..."
    git add .
    git commit -m "Final verification — all checks pass"
    git push
    git push hf main
    
    Write-Host "ALL PROCESSES COMPLETED SUCCESSFULLY"
} catch {
    Write-Host "ERROR ENCOUNTERED:"
    Write-Host $_
    exit 1
}

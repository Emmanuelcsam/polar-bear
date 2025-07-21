# Continuous Auto Git Push Script with Timer
# Save this as: auto-git-loop.ps1

# Configuration
$repoPath = "C:\Users\Saem1001\Documents\GitHub\polar-bear"
$intervalMinutes = 30  # Change this to your desired interval in minutes

# Convert to seconds
$intervalSeconds = $intervalMinutes * 60

Write-Host "Starting Auto Git Push Loop..." -ForegroundColor Cyan
Write-Host "Repository: $repoPath" -ForegroundColor Yellow
Write-Host "Interval: $intervalMinutes minutes" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

# Navigate to repository
Set-Location -Path $repoPath

# Check if we're in a git repository
if (!(Test-Path .git)) {
    Write-Host "Error: Not a git repository!" -ForegroundColor Red
    exit 1
}

# Continuous loop
while ($true) {
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "`n[$timestamp] Checking for changes..." -ForegroundColor Blue
    
    # Check for changes
    $status = git status --porcelain
    
    if ($status) {
        Write-Host "Changes detected! Committing and pushing..." -ForegroundColor Green
        
        # Stage all changes
        git add .
        
        # Commit with timestamp
        $commitMessage = "Auto commit: $timestamp"
        git commit -m $commitMessage
        
        # Push to remote
        git push
        
        Write-Host "Successfully pushed changes!" -ForegroundColor Green
        Write-Host "Commit message: $commitMessage" -ForegroundColor Cyan
        
        # Log the action
        $logPath = "$repoPath\auto-git-log.txt"
        Add-Content -Path $logPath -Value "[$timestamp] Changes pushed"
    } else {
        Write-Host "No changes to commit." -ForegroundColor Yellow
        
        # Log the check
        $logPath = "$repoPath\auto-git-log.txt"
        Add-Content -Path $logPath -Value "[$timestamp] No changes"
    }
    
    Write-Host "`nWaiting $intervalMinutes minutes until next check..." -ForegroundColor Gray
    Write-Host "Next check at: $((Get-Date).AddMinutes($intervalMinutes).ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Gray
    
    # Wait for the specified interval
    Start-Sleep -Seconds $intervalSeconds
}
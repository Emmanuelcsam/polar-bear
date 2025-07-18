# Auto Git Commit and Push Script for Windows
# Save this as: auto-git-push.ps1

# Configuration - CHANGE THESE VALUES
$repoPath = "C:\Users\Saem1001\Documents\GitHub\polar-bear"  # Change this to your repository path
$commitMessage = "Auto commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Navigate to repository
Set-Location -Path $repoPath

# Check if we're in a git repository
if (!(Test-Path .git)) {
    Write-Host "Error: Not a git repository!" -ForegroundColor Red
    exit 1
}

# Check for changes
$status = git status --porcelain

if ($status) {
    Write-Host "Changes detected. Committing and pushing..." -ForegroundColor Green
    
    # Stage all changes
    git add .
    
    # Commit with timestamp
    git commit -m $commitMessage
    
    # Push to remote
    git push
    
    Write-Host "Successfully committed and pushed changes!" -ForegroundColor Green
    Write-Host "Commit message: $commitMessage" -ForegroundColor Cyan
} else {
    Write-Host "No changes to commit." -ForegroundColor Yellow
}

# Log the action
$logPath = "$repoPath\auto-git-log.txt"
Add-Content -Path $logPath -Value "[$(Get-Date)] Script executed. Status: $(if($status){'Changes pushed'}else{'No changes'})"
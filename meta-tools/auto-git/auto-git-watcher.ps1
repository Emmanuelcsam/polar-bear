# Real-Time Git Auto Push File Watcher
# Automatically commits and pushes when files change

# Configuration
$repoPath = "C:\Users\Saem1001\Documents\GitHub\polar-bear"
$debounceSeconds = 5  # Wait 5 seconds after last change before committing

# Navigate to repository
Set-Location -Path $repoPath

# Check if we're in a git repository
if (!(Test-Path .git)) {
    Write-Host "Error: Not a git repository!" -ForegroundColor Red
    exit 1
}

Write-Host "Git Auto-Pusher Started!" -ForegroundColor Green
Write-Host "Watching: $repoPath" -ForegroundColor Yellow
Write-Host "Changes will be committed $debounceSeconds seconds after detected" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Gray

# Create a hashtable to track the timer
$global:debounceTimer = $null
$global:pendingCommit = $false

# Function to commit and push
function Invoke-GitPush {
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    
    Write-Host "`n[$timestamp] Committing and pushing changes..." -ForegroundColor Cyan
    
    # Check for actual changes
    $status = git status --porcelain
    
    if ($status) {
        # Stage all changes
        git add .
        
        # Commit with timestamp
        $commitMessage = "Auto commit: $timestamp"
        git commit -m $commitMessage
        
        # Push to remote
        git push
        
        Write-Host "Successfully pushed changes!" -ForegroundColor Green
        
        # Log the action
        $logPath = "$repoPath\auto-git-log.txt"
        Add-Content -Path $logPath -Value "[$timestamp] Changes pushed"
    } else {
        Write-Host "No changes to commit (files may have been deleted/renamed back)" -ForegroundColor Yellow
    }
    
    $global:pendingCommit = $false
}

# Create FileSystemWatcher
$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $repoPath
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

# Configure what to watch
$watcher.NotifyFilter = [System.IO.NotifyFilters]::FileName -bor 
                       [System.IO.NotifyFilters]::DirectoryName -bor
                       [System.IO.NotifyFilters]::LastWrite -bor
                       [System.IO.NotifyFilters]::Size

# Define the action for file changes
$action = {
    $path = $Event.SourceEventArgs.FullPath
    $changeType = $Event.SourceEventArgs.ChangeType
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    
    # Ignore .git directory and auto-git-log.txt
    if ($path -like "*\.git\*" -or $path -like "*\auto-git-log.txt") {
        return
    }
    
    # Get relative path for cleaner output
    $relativePath = $path.Replace("$repoPath\", "")
    
    Write-Host "[$timestamp] Detected: $changeType - $relativePath" -ForegroundColor DarkGray
    
    # Cancel existing timer if any
    if ($global:debounceTimer) {
        Unregister-Event -SourceIdentifier GitDebounceTimer -ErrorAction SilentlyContinue
        $global:debounceTimer = $null
    }
    
    # Set flag that we have pending changes
    $global:pendingCommit = $true
    
    # Create new timer to commit after debounce period
    $global:debounceTimer = Register-ObjectEvent -InputObject ([System.Timers.Timer]::new($debounceSeconds * 1000)) -EventName Elapsed -SourceIdentifier GitDebounceTimer -Action {
        if ($global:pendingCommit) {
            Invoke-GitPush
        }
        Unregister-Event -SourceIdentifier GitDebounceTimer
        $global:debounceTimer = $null
    }
    
    # Start the timer
    $global:debounceTimer.InputObject.AutoReset = $false
    $global:debounceTimer.InputObject.Start()
}

# Register event handlers
Register-ObjectEvent -InputObject $watcher -EventName "Changed" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Created" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Deleted" -Action $action
Register-ObjectEvent -InputObject $watcher -EventName "Renamed" -Action $action

# Initial status check
Write-Host "Initial repository status:" -ForegroundColor Blue
git status --short

try {
    # Keep the script running
    while ($true) {
        Start-Sleep -Seconds 1
    }
} finally {
    # Cleanup
    $watcher.EnableRaisingEvents = $false
    $watcher.Dispose()
    Write-Host "`nFile watcher stopped." -ForegroundColor Yellow
}
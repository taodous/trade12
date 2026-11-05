# Supprime la tâche planifiée du bot paper. Les positions du compte paper ne sont pas touchées.
#
# Usage : powershell -ExecutionPolicy Bypass -File scripts\uninstall_task.ps1

param(
    [string]$TaskName = "TradeBotPaper"
)

$ErrorActionPreference = "Stop"
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Tâche $TaskName supprimée."
} else {
    Write-Output "Aucune tâche $TaskName."
}

# Supprime les tâches planifiées du bot paper et de son chien de garde. Les positions du compte paper ne sont pas touchées.
#
# Usage : powershell -ExecutionPolicy Bypass -File scripts\uninstall_task.ps1

param(
    [string[]]$TaskNames = @("TradeBotPaper", "TradeBotWatchdog")
)

$ErrorActionPreference = "Stop"
foreach ($TaskName in $TaskNames) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Stop-ScheduledTask -TaskName $TaskName
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Tâche $TaskName supprimée."
    } else {
        Write-Output "Aucune tâche $TaskName."
    }
}

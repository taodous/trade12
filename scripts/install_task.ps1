# Installe la tâche planifiée qui lance le bot paper chaque jour de semaine.
#
# Le bot démarre avant l'ouverture, attend le marché, trade selon la stratégie puis s'arrête après
# la clôture (option --until-close). Un jour férié, il s'arrête aussitôt : le calendrier vient de
# l'horloge d'Alpaca. 14h15, heure de Paris, couvre les semaines de décalage d'heure d'été, où
# l'ouverture de New York est à 14h30 au lieu de 15h30.
#
# Usage : powershell -ExecutionPolicy Bypass -File scripts\install_task.ps1
# Relancer le script remplace la tâche existante.

param(
    [string]$TaskName = "TradeBotPaper",
    [string]$StartTime = "14:15",
    [int]$TimeLimitHours = 9
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$uv = (Get-Command uv).Source
$logs = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logs | Out-Null

# Les logs JSON vont dans logs\bot-AAAA-MM-JJ.jsonl (écrits par le bot). La sortie console est
# jetée, sauf les erreurs Python non prévues, gardées dans logs\stderr.log.
$command = "set PYTHONUTF8=1&& `"$uv`" run python -m bot.main --until-close 1>NUL 2>>`"$logs\stderr.log`""
# conhost --headless : aucune fenêtre console. Une fenêtre visible se ferme trop facilement, et
# la fermer arrête le bot en pleine séance (constaté le 2026-11-06).
$action = New-ScheduledTaskAction -Execute "conhost.exe" -Argument "--headless cmd.exe /d /c `"$command`"" -WorkingDirectory $root
$user = "$env:USERDOMAIN\$env:USERNAME"
$daily = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $StartTime
# Après un arrêt ou un redémarrage du PC en pleine séance, l'ouverture de session relance le bot,
# qui reprend la séance en cours. Un week-end ou après la clôture, il s'arrête aussitôt, et le
# journal empêche de traiter deux fois la même séance.
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours $TimeLimitHours) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries
# Session de l'utilisateur, sans élévation : aucun mot de passe n'est stocké. Le PC peut être
# verrouillé, mais la session doit rester ouverte.
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $daily, $logon `
    -Settings $settings `
    -Principal $principal `
    -Description "Bot de trading, compte PAPER Alpaca uniquement. Voir docs\paper_trading.md." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime, LastRunTime, LastTaskResult

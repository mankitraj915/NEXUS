$inputJson = $input | Out-String | ConvertFrom-Json

$reason = $inputJson.terminationReason

if ($reason -eq "model_stop") {
    $output = @{
        decision = "continue"
        reason = "Autonomous mode enabled: continuing original objective automatically."
    }
} else {
    $output = @{
        decision = "stop"
        reason = "Stopping due to $reason."
    }
}

$output | ConvertTo-Json -Compress | Write-Output

# Run All Experiments for SAC, DQN, PPO, and A2C
# Each with Discrete interpreter, Price data, Parametric and Bayesian

$agents = @("sac", "dqn", "ppo", "a2c")

$experiments = @()

foreach ($agent in $agents) {
    $exp_name = "${agent}_test"
    $params = "--agent-type $agent --interpreter-type discrete --price-type both --reward-projection-period 20 --indicator-type simple --reward-type log_returns"
    $experiments += @{name=$exp_name; params=$params}
}

Write-Host "Setting up $($experiments.Count) experiments..." -ForegroundColor Green

# Setup all experiments
foreach ($exp in $experiments) {
    $expName = "$($exp.name)"
    Write-Host "Setting up: $expName" -ForegroundColor Yellow
    $setupCmd = "python scripts/setup_experiment.py --experiment-name `"$expName`" $($exp.params)"
    Invoke-Expression $setupCmd
}

Write-Host "`nStarting training for all experiments..." -ForegroundColor Green

# Start training for all experiments
foreach ($exp in $experiments) {
    $expName = "$($exp.name)"
    Write-Host "Starting training: $expName" -ForegroundColor Yellow
    Start-Process -FilePath "python" -ArgumentList "scripts/start_experiment.py", "--experiment-name", "`"$expName`"" -NoNewWindow
    Start-Sleep -Seconds 2
}

Write-Host "`nAll experiments started!" -ForegroundColor Green
Write-Host "Check the experiments folder for logs and progress." -ForegroundColor Cyan 
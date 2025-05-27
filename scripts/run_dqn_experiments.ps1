# Simple DQN Experiments Runner
# Sets up and runs 8 DQN experiments simultaneously

# Define the 8 experiments
$experiments = @(
    @{name="dqn_discrete_price_parametric"; params="--agent-type dqn --interpreter-type discrete --price-type price"},
    @{name="dqn_discrete_price_bayesian"; params="--agent-type dqn --interpreter-type discrete --price-type price --use-bayesian"},
    @{name="dqn_confidence_price_parametric"; params="--agent-type dqn --interpreter-type confidence_scaled --price-type price"},
    @{name="dqn_confidence_price_bayesian"; params="--agent-type dqn --interpreter-type confidence_scaled --price-type price --use-bayesian"},
    @{name="dqn_discrete_ohlcv_parametric"; params="--agent-type dqn --interpreter-type discrete --price-type ohlcv"},
    @{name="dqn_discrete_ohlcv_bayesian"; params="--agent-type dqn --interpreter-type discrete --price-type ohlcv --use-bayesian"},
    @{name="dqn_confidence_ohlcv_parametric"; params="--agent-type dqn --interpreter-type confidence_scaled --price-type ohlcv"},
    @{name="dqn_confidence_ohlcv_bayesian"; params="--agent-type dqn --interpreter-type confidence_scaled --price-type ohlcv --use-bayesian"}
)

Write-Host "Setting up 8 DQN experiments..." -ForegroundColor Green

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
    
    # Start each experiment in background
    Start-Process -FilePath "python" -ArgumentList "scripts/start_experiment.py", "--experiment-name", "`"$expName`"" -NoNewWindow
    
    # Small delay between starts
    Start-Sleep -Seconds 2
}

Write-Host "`nAll experiments started!" -ForegroundColor Green
Write-Host "Check the experiments folder for logs and progress." -ForegroundColor Cyan 
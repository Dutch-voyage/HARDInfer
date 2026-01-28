# Guides on visualize latency breakdowns for different tasks

## Step 1: Collect stats
run 

```
uv run python ./eval/layouts_efficiency.py
```

## Step 2: Visualize
run 
```
uv run python ./viz/tasks_metrics.py
```

Then you can get bar figures like this:
![HS_task](figs/task_performance/HS_task_performance_32seqs.png)
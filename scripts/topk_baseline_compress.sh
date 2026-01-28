for data_source in umathtop50 aime24 aime25; do
  for layer_budget in 1024 2048 4096; do
      for window_size in 32; do
        for steps_between_cache_compressions in 128; do
          python -m eval.baseline \
              --data_source aime24 \
              --enforce_eager True \
              --compress_method vanilla \
              --layer_budget ${layer_budget} \
              --window_size ${window_size} \
              --steps_between_cache_compressions ${steps_between_cache_compressions}
        done
    done
  done
done
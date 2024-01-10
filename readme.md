Checkpoint merger with infinitely branching per-tensor merging with readable recipes stored in yaml.

Bullet points:
- Reduced memory usage by utilizing safetensors' lazy loading instead of having source checkpoints stored in memory.
- A cache that reuses time consuming calculations from recent merges
- Supports multithreading for a moderate performance improvement.
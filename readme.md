Checkpoint merger with infinitely branching per-tensor merging with recipes stored in yaml.

Bullet points:
- Capable of reusing calculations from recent merges to greatly improve merge times
- Supports multithreading for a moderate performance improvement.
- Lower memory usage (except for when it breaks and uses all memory)

Note:
- Only supports safetensors.

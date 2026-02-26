# System Design Examples

Python implementations of common system design patterns and components.

## Files

| File | Lesson | Description |
|------|--------|-------------|
| `04_load_balancer.py` | L04 | Round-robin, weighted, least-connections LB |
| `05_rate_limiter.py` | L05 | Token bucket, sliding window rate limiters |
| `06_cache_strategies.py` | L06 | Cache-aside, write-through, write-back patterns |
| `07_consistent_hashing.py` | L07 | Consistent hashing ring with virtual nodes |
| `08_sharding_sim.py` | L08 | Hash-based and range-based sharding |
| `10_eventual_consistency.py` | L10 | Vector clocks, LWW conflict resolution |
| `11_message_queue.py` | L11 | In-memory pub/sub message queue |
| `14_circuit_breaker.py` | L14 | Circuit breaker state machine |
| `14_saga_pattern.py` | L14 | Saga orchestrator with compensation |
| `16_raft_sim.py` | L16 | Raft consensus (leader election, log replication) |
| `17_url_shortener.py` | L17 | Base62 URL shortener |
| `19_metrics_collector.py` | L19 | Metrics with percentile calculation |
| `20_inverted_index.py` | L20 | Inverted index + BM25 scoring |

## Running

All examples use Python standard library only:

```bash
python 04_load_balancer.py
python 05_rate_limiter.py
python 07_consistent_hashing.py
python 16_raft_sim.py
```

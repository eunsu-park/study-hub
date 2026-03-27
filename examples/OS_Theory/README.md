# OS Theory Examples

Runnable Python simulators that illustrate operating system concepts.
All examples use only the Python standard library.

## Files

| File | Lesson | Description |
|------|--------|-------------|
| `02_process_demo.py` | L02 | Process creation, PCB, state transitions |
| `03_threading_demo.py` | L03 | Threads, race conditions, GIL |
| `05_scheduling_sim.py` | L05 | FCFS, SJF, SRTF, RR with Gantt charts |
| `06_mlfq_sim.py` | L06 | Multi-Level Feedback Queue |
| `07_sync_primitives.py` | L07 | Peterson's algorithm, mutex |
| `08_producer_consumer.py` | L08 | Producer-consumer, dining philosophers |
| `09_deadlock_detection.py` | L09 | Banker's algorithm, wait-for graph |
| `12_paging_sim.py` | L12 | Page table walk, TLB |
| `15_page_replacement.py` | L15 | FIFO, LRU, LFU, Clock |
| `17_filesystem_sim.py` | L17 | Inode-based filesystem, FAT |
| `18_ipc_demo.py` | L18 | Pipe, shared memory, message queue |

## Running

```bash
python 05_scheduling_sim.py   # CPU scheduling Gantt chart
python 15_page_replacement.py # Compare page replacement algorithms
python 09_deadlock_detection.py # Banker's algorithm demo
```

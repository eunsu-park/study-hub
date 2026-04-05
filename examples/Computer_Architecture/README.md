# Computer Architecture Examples

Python simulators demonstrating fundamental computer architecture concepts.

## Files

| File | Lesson | Description |
|------|--------|-------------|
| `02_number_systems.py` | L02 | Binary/hex/octal converter, two's complement |
| `03_ieee754.py` | L03 | IEEE 754 encoder/decoder, precision demonstration |
| `04_logic_gates.py` | L04 | Logic gate simulator (AND/OR/NOT/XOR/NAND/NOR) |
| `05_alu_adder.py` | L05 | Half/full adder, ripple-carry ALU |
| `06_flip_flops.py` | L06 | SR, D, JK flip-flop and register simulation |
| `07_cpu_datapath.py` | L07 | Single-cycle CPU datapath simulator |
| `10_assembly_sim.py` | L10 | Fetch-decode-execute ISA simulator |
| `11_pipeline_sim.py` | L11 | 5-stage pipeline with hazard detection |
| `12_branch_predictor.py` | L12 | 1-bit, 2-bit, BTB branch predictors |
| `15_cache_sim.py` | L15 | Direct-mapped/set-associative/fully-associative cache |
| `16_tlb_sim.py` | L16 | TLB + page table walk simulation |

## Running

All examples use Python standard library only:

```bash
python 02_number_systems.py
python 05_alu_adder.py
python 11_pipeline_sim.py
python 15_cache_sim.py
```

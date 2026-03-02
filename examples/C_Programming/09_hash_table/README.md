# Hash Table Project

A hash table project implementing various hash functions and collision resolution techniques.

## File Structure

### 1. hash_functions.c
A comparison and analysis tool for various hash functions

**Implemented Hash Functions:**
- `hash_simple` - Simple summation (many collisions, bad example)
- `hash_djb2` - Daniel J. Bernstein (recommended)
- `hash_sdbm` - sdbm database hash
- `hash_fnv1a` - Fowler-Noll-Vo 1a

**Features:**
- Hash value comparison output
- Collision count and collision rate analysis
- Distribution uniformity analysis (variance calculation)
- Hash distribution visualization

### 2. hash_chaining.c
Hash table implementation using separate chaining

**Characteristics:**
- Stores collisions in linked lists
- No table size limit
- Simple insertion/deletion

**Implemented Features:**
- `ht_create()` - Create hash table
- `ht_set()` - Insert/update
- `ht_get()` - Search
- `ht_delete()` - Delete
- `ht_print()` - Print table
- `ht_get_statistics()` - Collect statistics
- Chain length distribution visualization

### 3. hash_linear_probing.c
Open addressing hash table using linear probing

**Characteristics:**
- Searches for next empty slot on collision
- Good cache efficiency
- Deletion handled with DELETED state

**Implemented Features:**
- Three slot states (EMPTY, OCCUPIED, DELETED)
- Linear probing algorithm
- Clustering analysis
- Performance comparison by load factor
- Cluster visualization

### 4. dictionary.c
A practical dictionary program using hash tables

**Key Features:**
- Add/search/delete words
- Print full list
- File save/load (dictionary.txt)
- Search suggestions (partial matching)
- Search statistics and top 10 popular words
- Case-insensitive search

**Data Structure:**
- Chaining method
- Table size: 1000
- Search count tracking

## Compile and Run

### Individual Compilation
```bash
gcc -Wall -Wextra -std=c11 -o hash_functions hash_functions.c
gcc -Wall -Wextra -std=c11 -o hash_chaining hash_chaining.c
gcc -Wall -Wextra -std=c11 -o hash_linear_probing hash_linear_probing.c
gcc -Wall -Wextra -std=c11 -o dictionary dictionary.c
```

### Using Makefile
```bash
make                # Compile all programs
make hash_functions # Compile a specific program
make clean          # Delete generated files
make run_dict       # Run dictionary program
```

### Run
```bash
./hash_functions        # Hash function comparison
./hash_chaining         # Chaining test
./hash_linear_probing   # Linear probing test
./dictionary            # Dictionary program
```

## Learning Points

### Hash Function Selection
- **djb2**: General purpose, balanced performance
- **FNV-1a**: When fast speed is needed
- **sdbm**: For database purposes
- **Do not use Simple** (high collision rate)

### Collision Resolution Method Comparison

| Comparison Item | Chaining | Open Addressing |
|-----------------|----------|-----------------|
| Memory | Dynamic allocation | Fixed size |
| Deletion | Simple | Requires DELETED marker |
| Cache efficiency | Disadvantaged | Advantaged |
| Load factor | > 1 possible | < 1 required |
| Implementation complexity | Low | Medium |

### Time Complexity

| Operation | Average | Worst |
|-----------|---------|-------|
| Insert | O(1) | O(n) |
| Search | O(1) | O(n) |
| Delete | O(1) | O(n) |

### Performance Optimization Tips
1. Keep load factor at 0.7 or below
2. Choose a good hash function (djb2 recommended)
3. Use prime numbers for table size
4. Choose chaining vs open addressing based on use case

## Example Output

### hash_functions Execution Result
```
=== Hash Function Comparison ===

Key          | Simple | djb2 | sdbm | fnv1a
-------------|--------|------|------|------
apple        |     30 |   43 |   58 |    67
banana       |      9 |   42 |   49 |    52

=== Collision Analysis ===
Simple       |        14 | 28.0%
djb2         |         6 | 12.0%  <- Minimum collisions
```

### dictionary Usage Example
```
Selection: 1
Word: programming
Definition: programming; the task of writing instructions for a computer
✓ 'programming' added

Selection: 2
Word to search: prog
Words starting with 'prog':
  - programming
Total 1 found
```

## Extension Ideas

1. **Dynamic resizing**: Double table size when load factor exceeds 0.7
2. **Double hashing**: Determine probe interval with a second hash function
3. **Additional dictionary features**:
   - Example sentences
   - Phonetic notation
   - Synonyms/antonyms
4. **Performance measurement**: Measure actual execution time of each operation
5. **Multithreading**: Support read concurrency

## References

- Hash functions: [djb2, sdbm, FNV-1a algorithms](http://www.cse.yorku.ca/~oz/hash.html)
- Collision resolution: Cormen et al., "Introduction to Algorithms"
- Practical usage: Python `dict`, Java `HashMap`, C++ `unordered_map`

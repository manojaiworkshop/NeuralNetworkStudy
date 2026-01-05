# DETAILED MEMORY ALLOCATION EXPLANATION
## Stack vs Heap for Matrix Objects

## Understanding sizeof() Results

When you run the program, you see:
- **Stack Matrix**: `sizeof(stackMatrix)` = **40 bytes** (the actual Matrix object size)
- **Heap Matrix Pointer**: `sizeof(heapMatrix)` = **8 bytes** (just the pointer size on 64-bit system)

Let's break down WHY these sizes are what they are.

---

## CASE 1: Stack Allocation - 40 Bytes

```cpp
Matrix stackMatrix(2, 2);  // Object created on stack
cout << sizeof(stackMatrix);  // Output: 40 bytes
```

### Memory Layout of Matrix Object (40 bytes total)

```
Matrix Class Structure:
┌─────────────────────────────────────────────────────────────┐
│ class Matrix {                                              │
│     size_t rows;                     // 8 bytes (64-bit)    │
│     size_t cols;                     // 8 bytes (64-bit)    │
│     std::vector<std::vector<double>> data;  // 24 bytes     │
│ };                                                          │
└─────────────────────────────────────────────────────────────┘
```

### Breakdown of the 40 bytes:

#### 1. `size_t rows` - 8 bytes
```
┌──────────────┐
│ rows = 2     │  8 bytes (size_t on 64-bit system)
└──────────────┘
```

#### 2. `size_t cols` - 8 bytes
```
┌──────────────┐
│ cols = 2     │  8 bytes (size_t on 64-bit system)
└──────────────┘
```

#### 3. `std::vector<std::vector<double>> data` - 24 bytes
```
std::vector internal structure (24 bytes):
┌──────────────────────────────────────────┐
│ pointer to data       │  8 bytes         │ ← Points to actual array on HEAP
├──────────────────────────────────────────┤
│ size (current)        │  8 bytes         │ ← Number of elements (2 rows)
├──────────────────────────────────────────┤
│ capacity (allocated)  │  8 bytes         │ ← Allocated capacity
└──────────────────────────────────────────┘
```

**Total: 8 + 8 + 24 = 40 bytes**

---

## Complete Memory Picture for Stack Allocation

```
STACK (Fast, Limited Size):
┌─────────────────────────────────────────────────────────────┐
│ stackMatrix (40 bytes)                                      │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐                                         │
│ │ rows = 2        │  8 bytes                                │
│ ├─────────────────┤                                         │
│ │ cols = 2        │  8 bytes                                │
│ ├─────────────────┤                                         │
│ │ data (vector)   │  24 bytes                               │
│ │ ┌─────────────┐ │                                         │
│ │ │ ptr ────────┼─┼──┐                                      │
│ │ ├─────────────┤ │  │ Points to heap                      │
│ │ │ size = 2    │ │  │                                      │
│ │ ├─────────────┤ │  │                                      │
│ │ │ capacity=2  │ │  │                                      │
│ │ └─────────────┘ │  │                                      │
│ └─────────────────┘  │                                      │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       ↓
HEAP (Slow, Large Size):
┌──────────────────────┴──────────────────────────────────────┐
│ Outer vector (2 vector<double> objects)                     │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ vector[0] (inner vector)      24 bytes               │   │
│ │ ┌──────────┬──────────┬──────────┐                   │   │
│ │ │ ptr ─────┼→ [1.0][2.0]  (16 bytes on heap)         │   │
│ │ │ size = 2 │  8 bytes │                              │   │
│ │ │ cap = 2  │  8 bytes │                              │   │
│ │ └──────────┴──────────┴──────────┘                   │   │
│ └───────────────────────────────────────────────────────┘   │
│ ┌───────────────────────────────────────────────────────┐   │
│ │ vector[1] (inner vector)      24 bytes               │   │
│ │ ┌──────────┬──────────┬──────────┐                   │   │
│ │ │ ptr ─────┼→ [3.0][4.0]  (16 bytes on heap)         │   │
│ │ │ size = 2 │  8 bytes │                              │   │
│ │ │ cap = 2  │  8 bytes │                              │   │
│ │ └──────────┴──────────┴──────────┘                   │   │
│ └───────────────────────────────────────────────────────┘   │
│                                                             │
│ Total on heap:                                              │
│ - Outer vector: 2 × 24 bytes = 48 bytes                     │
│ - Inner data: 2 × 16 bytes = 32 bytes                       │
│ - Total: 80 bytes (approximately)                           │
└─────────────────────────────────────────────────────────────┘
```

### Why 40 bytes on stack?
- The Matrix **object itself** (rows, cols, vector structure) = **40 bytes**
- The actual **data** (the numbers 1.0, 2.0, 3.0, 4.0) lives on **heap** (~80 bytes)

---

## CASE 2: Heap Allocation - 8 Bytes (Pointer)

```cpp
Matrix* heapMatrix = new Matrix(2, 2);  // Object created on heap
cout << sizeof(heapMatrix);  // Output: 8 bytes (pointer size)
```

### Memory Layout

```
STACK:
┌─────────────────────────────────────────────────────────────┐
│ heapMatrix (pointer)                                        │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────────────┐                                    │
│ │ address = 0x55abc... │  8 bytes (64-bit pointer)         │
│ └──────────────────────┘                                    │
│              │                                              │
│              │ Points to object on heap                     │
└──────────────┼──────────────────────────────────────────────┘
               │
               ↓
HEAP (Object):
┌──────────────┴──────────────────────────────────────────────┐
│ Matrix object (40 bytes) at address 0x55abc...              │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐                                         │
│ │ rows = 2        │  8 bytes                                │
│ ├─────────────────┤                                         │
│ │ cols = 2        │  8 bytes                                │
│ ├─────────────────┤                                         │
│ │ data (vector)   │  24 bytes                               │
│ │ ┌─────────────┐ │                                         │
│ │ │ ptr ────────┼─┼──┐                                      │
│ │ ├─────────────┤ │  │                                      │
│ │ │ size = 2    │ │  │                                      │
│ │ ├─────────────┤ │  │                                      │
│ │ │ capacity=2  │ │  │                                      │
│ │ └─────────────┘ │  │                                      │
│ └─────────────────┘  │                                      │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       ↓
HEAP (Data):
┌──────────────────────┴──────────────────────────────────────┐
│ Actual matrix data (same as before)                         │
│ - Outer vector: 48 bytes                                    │
│ - Inner data: 32 bytes                                      │
│ - Total: ~80 bytes                                          │
└─────────────────────────────────────────────────────────────┘
```

### Why only 8 bytes for pointer?
- `heapMatrix` is just a **pointer** variable stored on the stack
- Pointers on 64-bit systems are **always 8 bytes** (memory address)
- The actual Matrix object (40 bytes) is on the **heap**
- The matrix data (~80 bytes) is also on the **heap**

---

## Complete Comparison

### Stack Allocation: `Matrix stackMatrix(2, 2);`

```
Memory Distribution:
┌─────────────┬──────────────┬───────────┐
│ Location    │ What         │ Size      │
├─────────────┼──────────────┼───────────┤
│ STACK       │ Matrix obj   │ 40 bytes  │
│ HEAP        │ Matrix data  │ ~80 bytes │
├─────────────┴──────────────┴───────────┤
│ Total: 120 bytes                       │
└────────────────────────────────────────┘

sizeof(stackMatrix) = 40 bytes
(only the object on stack, not the data on heap)
```

### Heap Allocation: `Matrix* heapMatrix = new Matrix(2, 2);`

```
Memory Distribution:
┌─────────────┬──────────────┬───────────┐
│ Location    │ What         │ Size      │
├─────────────┼──────────────┼───────────┤
│ STACK       │ Pointer      │ 8 bytes   │
│ HEAP        │ Matrix obj   │ 40 bytes  │
│ HEAP        │ Matrix data  │ ~80 bytes │
├─────────────┴──────────────┴───────────┤
│ Total: 128 bytes                       │
└────────────────────────────────────────┘

sizeof(heapMatrix) = 8 bytes
(only the pointer on stack, not the object or data)
```

---

## Detailed Size Calculation

### Why is size_t 8 bytes?

```cpp
size_t rows;  // 8 bytes on 64-bit system
size_t cols;  // 8 bytes on 64-bit system
```

- On **32-bit systems**: size_t = 4 bytes
- On **64-bit systems**: size_t = 8 bytes
- size_t is designed to hold any memory address
- Your system is 64-bit, so size_t = 8 bytes

### Why is std::vector 24 bytes?

```cpp
std::vector<std::vector<double>> data;  // 24 bytes
```

**Internal structure of std::vector:**
```cpp
template<typename T>
class vector {
    T* _data;        // 8 bytes: pointer to heap array
    size_t _size;    // 8 bytes: current number of elements
    size_t _capacity; // 8 bytes: allocated capacity
    // Total: 24 bytes
};
```

This 24 bytes is just the "control structure" - the actual data is elsewhere on the heap!

---

## Visual Comparison: Address Space

```
HIGH ADDRESS (Stack grows down)
┌─────────────────────────────────────────┐
│         STACK MEMORY                    │
├─────────────────────────────────────────┤
│                                         │
│  For Stack Allocation:                 │
│  ┌────────────────────────────────┐    │
│  │ stackMatrix (40 bytes)         │    │ ← sizeof = 40
│  └────────────────────────────────┘    │
│                                         │
│  For Heap Allocation:                  │
│  ┌────────────────────────────────┐    │
│  │ heapMatrix (8 bytes pointer)   │    │ ← sizeof = 8
│  └────────────────────────────────┘    │
│                                         │
├─────────────────────────────────────────┤
│       FREE MEMORY (unused)              │
├─────────────────────────────────────────┤
│         HEAP MEMORY                     │
│                                         │
│  ┌────────────────────────────────┐    │
│  │ Matrix object (40 bytes)       │    │ ← new Matrix
│  │ - if heap allocated            │    │
│  └────────────────────────────────┘    │
│                                         │
│  ┌────────────────────────────────┐    │
│  │ vector data (~80 bytes)        │    │ ← Actual numbers
│  │ - for both cases               │    │
│  └────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
LOW ADDRESS (Heap grows up)
```

---

## Key Takeaways

1. **sizeof() measures the object size, not pointed-to data**
   - `sizeof(stackMatrix)` = 40 bytes (Matrix object structure)
   - `sizeof(heapMatrix)` = 8 bytes (just the pointer)

2. **Matrix object = 40 bytes because:**
   - rows (size_t) = 8 bytes
   - cols (size_t) = 8 bytes
   - vector structure = 24 bytes
   - Total: 8 + 8 + 24 = 40 bytes

3. **Pointer = 8 bytes because:**
   - 64-bit system uses 8-byte addresses
   - Pointers store memory addresses
   - All pointers are 8 bytes on 64-bit (whether pointing to int, Matrix, or anything)

4. **The actual matrix data lives on heap in BOTH cases**
   - std::vector always allocates its data on the heap
   - The 40 bytes is just the "wrapper" object
   - The real data (1.0, 2.0, 3.0, 4.0) is ~80 bytes on heap

5. **Total memory used:**
   - Stack Matrix: 40 (stack) + 80 (heap) = 120 bytes
   - Heap Matrix: 8 (stack) + 40 (heap) + 80 (heap) = 128 bytes

---

## Practical Example with Addresses

When you run the program, you might see:

```
Stack Matrix address: 0x7ffc8a2b1c40    ← High address (stack)
Heap Matrix address:  0x55c8b2e01e20    ← Low address (heap)
```

This confirms:
- Stack grows from high to low addresses
- Heap grows from low to high addresses
- They are in completely different memory regions!

---

This is why understanding sizeof() is crucial - it only measures the object itself, not any dynamically allocated memory it points to!

# Memory Management Guide: Stack vs Heap in C++

## Table of Contents
1. [Memory Layout in C++](#memory-layout)
2. [Stack Memory](#stack-memory)
3. [Heap Memory](#heap-memory)
4. [Matrix Example: Memory Analysis](#matrix-memory)
5. [Variable Storage and Manipulation](#variable-storage)
6. [Performance Considerations](#performance)
7. [Best Practices](#best-practices)

---

## 1. Memory Layout in C++ {#memory-layout}

When your program runs, memory is divided into segments:

```
High Address
┌─────────────────┐
│   Stack         │  ← Local variables, function calls
│   ↓ grows down  │
├─────────────────┤
│                 │
│   Free Memory   │
│                 │
├─────────────────┤
│   ↑ grows up    │
│   Heap          │  ← Dynamic allocations (new/malloc)
├─────────────────┤
│   BSS Segment   │  ← Uninitialized global/static variables
├─────────────────┤
│   Data Segment  │  ← Initialized global/static variables
├─────────────────┤
│   Code/Text     │  ← Your compiled program code
└─────────────────┘
Low Address
```

---

## 2. Stack Memory {#stack-memory}

### What is Stack Memory?

The stack is a **LIFO (Last In First Out)** memory region automatically managed by the compiler.

### Characteristics

| Feature | Stack |
|---------|-------|
| **Allocation** | Automatic |
| **Deallocation** | Automatic (when scope ends) |
| **Speed** | Very fast (just move stack pointer) |
| **Size** | Limited (typically 1-8 MB) |
| **Lifetime** | Function scope |
| **Fragmentation** | None |
| **Access** | Sequential |

### Stack Example

```cpp
void function() {
    int x = 5;           // Allocated on stack
    double y = 3.14;     // Allocated on stack
    Matrix m1(2, 2);     // Matrix object on stack
    
    // When function ends, ALL these are automatically freed
}
```

### Stack Frame Visualization

```cpp
void foo() {
    int a = 10;
    bar(a);
}

void bar(int x) {
    int y = x * 2;
    // Stack at this point:
}
```

**Stack during `bar()` execution:**
```
Stack Pointer (SP) →  ┌──────────────┐
                      │ y = 20       │  ← bar's local variable
                      ├──────────────┤
                      │ x = 10       │  ← bar's parameter
                      ├──────────────┤
                      │ Return addr  │  ← Where to return after bar()
                      ├──────────────┤
                      │ a = 10       │  ← foo's local variable
                      ├──────────────┤
                      │ Return addr  │  ← Where to return after foo()
                      └──────────────┘
```

### Stack Overflow

Stack overflow occurs when you exceed stack size:

```cpp
// BAD: Too large for stack
void riskyFunction() {
    double bigArray[10000000];  // ~76 MB - will crash!
}

// GOOD: Use heap instead
void safeFunction() {
    double* bigArray = new double[10000000];  // Heap allocation
    // ... use array ...
    delete[] bigArray;
}
```

---

## 3. Heap Memory {#heap-memory}

### What is Heap Memory?

The heap is a large pool of memory for **dynamic allocation** that you manually manage.

### Characteristics

| Feature | Heap |
|---------|------|
| **Allocation** | Manual (new/malloc) |
| **Deallocation** | Manual (delete/free) |
| **Speed** | Slower (complex allocation algorithm) |
| **Size** | Large (limited by system RAM) |
| **Lifetime** | Until you free it |
| **Fragmentation** | Possible |
| **Access** | Random access via pointers |

### Heap Example

```cpp
void function() {
    int* p = new int(5);           // Allocated on heap
    double* arr = new double[100]; // Array on heap
    Matrix* m = new Matrix(2, 2);  // Matrix on heap
    
    // Use the memory...
    
    // MUST manually free
    delete p;
    delete[] arr;
    delete m;
}
```

### Heap Allocation Process

```cpp
int* ptr = new int(42);
```

**What happens internally:**

1. **Request memory from heap manager**
   - Heap manager searches free blocks
   - Finds suitable block (or requests from OS)

2. **Memory allocation**
   ```
   Heap Before:
   ┌─────────┬──────┬─────────┐
   │ Used    │ Free │ Used    │
   └─────────┴──────┴─────────┘
   
   Heap After:
   ┌─────────┬──────┬─────────┐
   │ Used    │ ptr→ │ Used    │
   └─────────┴──────┴─────────┘
                 ↑
              4 bytes for int
   ```

3. **Constructor called** (for objects)
4. **Pointer returned** to your variable

### Memory Leaks

```cpp
// BAD: Memory leak
void leakyFunction() {
    int* p = new int(10);
    // Function ends, 'p' is destroyed, but memory still allocated!
    // LEAKED 4 bytes
}

// GOOD: Proper cleanup
void properFunction() {
    int* p = new int(10);
    // ... use p ...
    delete p;  // Free the memory
}
```

**After 1000 calls to `leakyFunction()`:**
- Lost 4000 bytes (4 KB)
- Memory still allocated but unreachable
- Program memory usage grows

---

## 4. Matrix Example: Memory Analysis {#matrix-memory}

Let's analyze our Matrix class memory usage in detail.

### Matrix Class Structure

```cpp
class Matrix {
private:
    int rows;        // 4 bytes on stack (inside object)
    int cols;        // 4 bytes on stack (inside object)
    double* data;    // 8 bytes pointer on stack (inside object)
                     // Points to heap memory for actual data
};
```

### Example 1: Matrix on Stack

```cpp
void example1() {
    Matrix m1(2, 3);  // Matrix object on stack
}
```

**Memory Layout:**

```
STACK:
┌─────────────────────────┐
│ m1 (Matrix object)      │
├─────────────────────────┤
│ rows = 2    (4 bytes)   │
│ cols = 3    (4 bytes)   │
│ data = 0x... (8 bytes)  │ ────┐
└─────────────────────────┘     │
        16 bytes total          │ Points to ↓

HEAP:
┌─────────────────────────┐
│ data[0] = 0.0 (8 bytes) │ ←───┘
│ data[1] = 0.0 (8 bytes) │
│ data[2] = 0.0 (8 bytes) │
│ data[3] = 0.0 (8 bytes) │
│ data[4] = 0.0 (8 bytes) │
│ data[5] = 0.0 (8 bytes) │
└─────────────────────────┘
     48 bytes (6 doubles)

Total Memory: 16 bytes (stack) + 48 bytes (heap) = 64 bytes
```

**Constructor Execution:**
```cpp
Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data = new double[rows * cols];  // Heap allocation!
    // Allocates: 2 * 3 * 8 = 48 bytes on heap
    
    for(int i = 0; i < rows * cols; i++) {
        data[i] = 0.0;  // Initialize heap memory
    }
}
```

**Destructor Execution:**
```cpp
Matrix::~Matrix() {
    delete[] data;  // Free heap memory
}
// After destructor: heap memory freed, stack frame removed
```

### Example 2: Matrix Pointer on Stack

```cpp
void example2() {
    Matrix* m1 = new Matrix(2, 3);  // Pointer on stack, object on heap
}
```

**Memory Layout:**

```
STACK:
┌─────────────────────────┐
│ m1 = 0x... (8 bytes)    │ ────┐ Points to ↓
└─────────────────────────┘     │

HEAP (Matrix Object):
┌─────────────────────────┐     │
│ rows = 2    (4 bytes)   │ ←───┘
│ cols = 3    (4 bytes)   │
│ data = 0x... (8 bytes)  │ ────┐
└─────────────────────────┘     │
        16 bytes                │ Points to ↓

HEAP (Matrix Data):
┌─────────────────────────┐     │
│ data[0] = 0.0 (8 bytes) │ ←───┘
│ data[1] = 0.0 (8 bytes) │
│ ... (6 doubles total)   │
└─────────────────────────┘
     48 bytes

Total Memory: 8 bytes (stack) + 16 bytes (heap object) + 48 bytes (heap data) = 72 bytes
```

**Cleanup Required:**
```cpp
delete m1;  // Calls destructor, frees heap data AND object
```

### Example 3: Matrix Addition Memory Flow

```cpp
Matrix m1(2, 2);  // Stack object, heap data
Matrix m2(2, 2);  // Stack object, heap data

m1.set(0, 0, 1.0);
m1.set(0, 1, 2.0);
m2.set(0, 0, 5.0);
m2.set(0, 1, 6.0);

Matrix result = m1.add(m2);  // New matrix returned
```

**Memory at each step:**

**Step 1: Create m1**
```
Stack: m1 object (16 bytes)
Heap:  m1.data (32 bytes for 4 doubles)
```

**Step 2: Create m2**
```
Stack: m1 object (16 bytes) + m2 object (16 bytes)
Heap:  m1.data (32 bytes) + m2.data (32 bytes)
Total: 96 bytes
```

**Step 3: Call m1.set(0, 0, 1.0)**
```cpp
void Matrix::set(int row, int col, double value) {
    data[row * cols + col] = value;  // Writes to heap memory
    // row=0, col=0, cols=2: data[0] = 1.0
}
```
```
Heap (m1.data):
┌──────┬──────┬──────┬──────┐
│ 1.0  │ 0.0  │ 0.0  │ 0.0  │  ← Modified
└──────┴──────┴──────┴──────┘
```

**Step 4: Call m1.add(m2)**
```cpp
Matrix Matrix::add(const Matrix& other) const {
    Matrix result(rows, cols);  // New matrix created
    // Stack: result object (16 bytes)
    // Heap: result.data (32 bytes)
    
    for(int i = 0; i < rows * cols; i++) {
        result.data[i] = this->data[i] + other.data[i];
    }
    
    return result;  // Copy returned (or move in modern C++)
}
```

**During add() execution:**
```
Stack: 
  - m1 (16 bytes)
  - m2 (16 bytes)
  - result (16 bytes) ← temporary

Heap:
  - m1.data (32 bytes) [1.0, 2.0, 0.0, 0.0]
  - m2.data (32 bytes) [5.0, 6.0, 0.0, 0.0]
  - result.data (32 bytes) [6.0, 8.0, 0.0, 0.0] ← computed

Total: 144 bytes
```

**Step 5: Assignment to result**
```cpp
Matrix result = m1.add(m2);
```

In modern C++ (C++11+), **move semantics** avoid copying:
```
1. add() creates temporary result
2. Return moves result (no copy!)
3. Temporary's data pointer transferred
4. No heap allocation/deallocation
```

**Final memory:**
```
Stack: m1 (16), m2 (16), result (16) = 48 bytes
Heap: m1.data (32), m2.data (32), result.data (32) = 96 bytes
Total: 144 bytes
```

---

## 5. Variable Storage and Manipulation {#variable-storage}

### Local Variables (Stack)

```cpp
void foo() {
    int x = 10;          // x on stack
    double y = 3.14;     // y on stack
    
    // Memory addresses
    cout << "Address of x: " << &x << endl;  // Stack address
    cout << "Address of y: " << &y << endl;  // Stack address
}
```

**Stack frame:**
```
High Address
┌──────────────┐
│ y = 3.14     │ ← 0x7ffc1234
├──────────────┤
│ x = 10       │ ← 0x7ffc1230
└──────────────┘
Low Address
```

### Dynamic Variables (Heap)

```cpp
void bar() {
    int* p = new int(10);      // p on stack, *p on heap
    double* q = new double(3.14);  // q on stack, *q on heap
    
    cout << "Address of p: " << &p << endl;    // Stack address
    cout << "Value of p: " << p << endl;       // Heap address
    cout << "Value of *p: " << *p << endl;     // Value in heap
    
    delete p;
    delete q;
}
```

**Memory layout:**
```
STACK:
┌──────────────────┐
│ q = 0x55b0...    │ ← Pointer (8 bytes)
├──────────────────┤
│ p = 0x55b0...    │ ← Pointer (8 bytes)
└──────────────────┘

HEAP:
┌──────────────────┐
│ *q = 3.14        │ ← 0x55b0... (8 bytes)
├──────────────────┤
│ *p = 10          │ ← 0x55b0... (4 bytes)
└──────────────────┘
```

### Arrays

**Stack Array:**
```cpp
int arr[100];  // 400 bytes on stack
// Fast, automatic cleanup
// Size must be compile-time constant
```

**Heap Array:**
```cpp
int* arr = new int[100];  // 400 bytes on heap
// Slower, manual cleanup
// Size can be runtime variable
delete[] arr;
```

### Matrix Operations: Step-by-Step Memory

```cpp
Matrix m1(2, 2);
```
**Stack:** Create m1 object (rows=2, cols=2, data=nullptr initially)
**Heap:** Allocate 32 bytes for 4 doubles

```cpp
m1.set(0, 0, 5.0);
```
**Operation:** `data[0*2 + 0] = 5.0` → Write to heap address

```cpp
double val = m1.get(0, 0);
```
**Operation:** `return data[0*2 + 0]` → Read from heap address → Copy to stack variable `val`

```cpp
Matrix m2 = m1;
```
**Default Copy (if no copy constructor):**
- Shallow copy: m2.data = m1.data (DANGER! Both point to same heap memory)

**Our Copy Constructor (proper):**
```cpp
Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
    data = new double[rows * cols];  // Allocate NEW heap memory
    for(int i = 0; i < rows * cols; i++) {
        data[i] = other.data[i];  // Copy values
    }
}
```
**Result:** Deep copy - separate heap memory for m2

---

## 6. Performance Considerations {#performance}

### Stack vs Heap Performance

| Operation | Stack | Heap |
|-----------|-------|------|
| Allocation | ~1 instruction | ~100+ instructions |
| Deallocation | ~1 instruction | ~100+ instructions |
| Access Speed | Very fast (cache-friendly) | Slower (random access) |
| Fragmentation | None | Possible |

### Benchmark Example

```cpp
#include <chrono>

// Stack allocation benchmark
void stackBenchmark() {
    auto start = chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 1000000; i++) {
        int x = 42;  // Stack allocation
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Stack: " << duration.count() << " μs" << endl;
}

// Heap allocation benchmark
void heapBenchmark() {
    auto start = chrono::high_resolution_clock::now();
    
    for(int i = 0; i < 1000000; i++) {
        int* x = new int(42);  // Heap allocation
        delete x;
    }
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Heap: " << duration.count() << " μs" << endl;
}
```

**Typical results:**
- Stack: ~2 ms
- Heap: ~200 ms (100x slower!)

### Matrix Performance Tips

**1. Reserve capacity if size known**
```cpp
// Instead of growing dynamically
vector<Matrix> matrices;
matrices.reserve(1000);  // Pre-allocate to avoid reallocations
```

**2. Pass by reference to avoid copies**
```cpp
// BAD: Copies entire matrix
void process(Matrix m) { }

// GOOD: No copy, read-only
void process(const Matrix& m) { }

// GOOD: No copy, can modify
void process(Matrix& m) { }
```

**3. Use move semantics**
```cpp
Matrix createMatrix() {
    Matrix m(1000, 1000);  // Large matrix
    // ... fill with data ...
    return m;  // Moved, not copied!
}
```

**4. Consider memory pools for many small allocations**
```cpp
// For neural networks with many small matrices
class MatrixPool {
    vector<double*> pool;
    // Reuse memory instead of new/delete
};
```

---

## 7. Best Practices {#best-practices}

### Stack: When to Use

✅ **Use stack for:**
- Small objects (< 1KB)
- Short-lived variables
- Function parameters
- Return values (with move semantics)

```cpp
void goodStackUsage() {
    Matrix m(2, 2);      // Small matrix - OK on stack
    int count = 0;       // Simple variable
    double sum = 0.0;    // Simple variable
}
```

### Heap: When to Use

✅ **Use heap for:**
- Large objects (> 1KB)
- Dynamic size (unknown at compile time)
- Long lifetime (beyond function scope)
- Polymorphism (base class pointers)

```cpp
Matrix* createLargeMatrix(int size) {
    return new Matrix(size, size);  // Large, return to caller
}
```

### Memory Management Rules

**Rule 1: Match new with delete**
```cpp
int* p = new int;          delete p;
int* arr = new int[10];    delete[] arr;  // Note: delete[]
Matrix* m = new Matrix();  delete m;
```

**Rule 2: Use smart pointers (modern C++)**
```cpp
#include <memory>

// Automatic cleanup!
auto m = std::make_unique<Matrix>(10, 10);
// No need to delete - freed automatically

// Shared ownership
auto m2 = std::make_shared<Matrix>(5, 5);
// Reference counted - freed when last reference gone
```

**Rule 3: Follow RAII (Resource Acquisition Is Initialization)**
```cpp
class Matrix {
public:
    Matrix(int r, int c) {
        data = new double[r * c];  // Acquire in constructor
    }
    
    ~Matrix() {
        delete[] data;  // Release in destructor
    }
};

// Usage: automatic cleanup
{
    Matrix m(5, 5);  // Constructor allocates
    // Use matrix...
}  // Destructor frees - no manual delete needed!
```

### Common Mistakes to Avoid

❌ **Mistake 1: Memory leak**
```cpp
void leak() {
    Matrix* m = new Matrix(10, 10);
    // Forgot delete!
}
```
✅ **Fix:** Use smart pointers or delete manually

❌ **Mistake 2: Double free**
```cpp
int* p = new int(5);
delete p;
delete p;  // ERROR: Already freed!
```
✅ **Fix:** Set pointer to nullptr after delete
```cpp
delete p;
p = nullptr;
```

❌ **Mistake 3: Use after free**
```cpp
int* p = new int(5);
delete p;
cout << *p;  // ERROR: Accessing freed memory!
```
✅ **Fix:** Don't use pointer after deletion

❌ **Mistake 4: Stack overflow**
```cpp
void overflow() {
    double bigArray[10000000];  // Too large for stack!
}
```
✅ **Fix:** Use heap for large arrays
```cpp
void ok() {
    double* bigArray = new double[10000000];
    // ... use ...
    delete[] bigArray;
}
```

### Debugging Memory Issues

**Using Valgrind:**
```bash
valgrind --leak-check=full --show-leak-kinds=all ./matrix_example
```

**Output interpretation:**
```
LEAK SUMMARY:
   definitely lost: 48 bytes in 1 blocks    ← Memory leaked
   indirectly lost: 0 bytes in 0 blocks
   possibly lost: 0 bytes in 0 blocks
   still reachable: 0 bytes in 0 blocks
```

**Using AddressSanitizer:**
```bash
g++ -fsanitize=address -g matrix.cpp -o matrix
./matrix
```

**Catches:**
- Heap buffer overflow
- Stack buffer overflow  
- Use after free
- Use after return
- Memory leaks

---

## Summary: Stack vs Heap

| Aspect | Stack | Heap |
|--------|-------|------|
| **Speed** | ⚡ Very Fast | 🐌 Slower |
| **Size** | 📦 Limited (~8MB) | 💾 Large (GB) |
| **Management** | 🤖 Automatic | 👤 Manual |
| **Lifetime** | ⏱️ Function scope | ♾️ Until freed |
| **Fragmentation** | ✅ None | ⚠️ Possible |
| **Cache** | 🎯 Cache-friendly | 💥 Cache-unfriendly |
| **Errors** | Stack overflow | Memory leaks, use-after-free |

### Golden Rules

1. **Prefer stack when possible** - faster, safer
2. **Use heap for large/dynamic data** - necessary for flexibility
3. **Always free what you allocate** - no leaks
4. **Use RAII pattern** - automatic resource management
5. **Consider smart pointers** - modern C++ best practice

This understanding is crucial for building efficient neural networks where memory management can significantly impact performance!

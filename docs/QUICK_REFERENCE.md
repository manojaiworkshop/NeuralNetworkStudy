# Quick Reference Guide for C++ Neural Network Project

## Quick Start Commands

### Building the Project

```bash
# First time build
mkdir build
cd build
cmake ..
make

# Subsequent builds (after code changes)
cd build
make

# Clean rebuild
rm -rf build
mkdir build
cd build
cmake ..
make
```

### Running the Program

```bash
# From project root
./build/matrix_example

# From build directory
cd build
./matrix_example
```

### Using the Build Script

```bash
# Make executable (first time only)
chmod +x build.sh

# Build and run
./build.sh
```

---

## CMake Quick Reference

### Common CMake Commands

```bash
# Generate build files (Debug mode)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Generate build files (Release mode)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build the project
cmake --build .

# Build specific target
cmake --build . --target matrix_example

# Clean build artifacts
cmake --build . --target clean

# Verbose build (see all commands)
cmake --build . --verbose
# or
make VERBOSE=1
```

### CMake Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CMAKE_CXX_STANDARD` | C++ version | `set(CMAKE_CXX_STANDARD 17)` |
| `CMAKE_BUILD_TYPE` | Debug/Release | `set(CMAKE_BUILD_TYPE Debug)` |
| `CMAKE_CXX_FLAGS` | Compiler flags | `set(CMAKE_CXX_FLAGS "-Wall")` |
| `PROJECT_SOURCE_DIR` | Project root path | `${PROJECT_SOURCE_DIR}/include` |
| `PROJECT_BINARY_DIR` | Build directory | `${PROJECT_BINARY_DIR}` |

### CMake Functions

```cmake
# Create executable
add_executable(name source1.cpp source2.cpp)

# Create library
add_library(name STATIC source1.cpp source2.cpp)

# Link library to target
target_link_libraries(target library1 library2)

# Add include directories
include_directories(path1 path2)
# or (modern)
target_include_directories(target PUBLIC path)

# Find source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB_RECURSE SOURCES "src/*.cpp")  # Include subdirectories

# Add subdirectory
add_subdirectory(subfolder)
```

---

## GDB Debugging Quick Reference

### Starting GDB

```bash
# Start with program
gdb ./matrix_example

# Start with core dump
gdb ./matrix_example core

# Attach to running process
gdb -p <pid>
```

### Essential GDB Commands

#### Running & Stopping

```gdb
run [args]              # Start program
continue (c)            # Continue execution
quit (q)                # Exit GDB
kill                    # Kill running program
```

#### Breakpoints

```gdb
break main              # Break at function
break file.cpp:42       # Break at line
break Class::method     # Break at method
info breakpoints        # List breakpoints
delete 1                # Delete breakpoint 1
disable 1               # Disable breakpoint 1
enable 1                # Enable breakpoint 1
clear                   # Delete all breakpoints
```

#### Stepping

```gdb
step (s)                # Step into function
next (n)                # Step over function
finish                  # Finish current function
until                   # Run until line
continue (c)            # Continue to next breakpoint
```

#### Examining Variables

```gdb
print var               # Print variable
print *ptr              # Dereference pointer
print arr[5]            # Array element
print obj.member        # Object member
display var             # Auto-print at each step
info locals             # Show local variables
info args               # Show function arguments
```

#### Examining Memory

```gdb
x/4i $pc                # Show 4 instructions at PC
x/10x ptr               # Show 10 hex values at ptr
x/s ptr                 # Show string at ptr
info registers          # Show CPU registers
backtrace (bt)          # Show call stack
frame 2                 # Switch to frame 2
up                      # Move up call stack
down                    # Move down call stack
```

#### Watchpoints

```gdb
watch var               # Break when var changes
watch *0x12345678       # Break when memory changes
rwatch var              # Break when var is read
awatch var              # Break on read/write
info watchpoints        # List watchpoints
```

### GDB Example Session

```bash
$ gdb ./matrix_example
(gdb) break Matrix::add
Breakpoint 1 at matrix.cpp:50

(gdb) run
Starting program: ./matrix_example
Breakpoint 1, Matrix::add

(gdb) print this->rows
$1 = 2

(gdb) print this->data[0]
$2 = 1.0

(gdb) next
(gdb) next
(gdb) print result.data[0]
$3 = 6.0

(gdb) continue
Program exited normally.

(gdb) quit
```

---

## Valgrind Memory Checking

### Basic Usage

```bash
# Memory leak detection
valgrind --leak-check=full ./matrix_example

# Detailed leak information
valgrind --leak-check=full --show-leak-kinds=all ./matrix_example

# Track uninitialized values
valgrind --track-origins=yes ./matrix_example

# Full reporting
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose ./matrix_example
```

### Understanding Output

```
LEAK SUMMARY:
   definitely lost: 48 bytes in 1 blocks     # Must fix: clear memory leak
   indirectly lost: 96 bytes in 2 blocks     # Lost via lost pointers
   possibly lost: 24 bytes in 1 blocks       # Uncertain (interior pointers)
   still reachable: 0 bytes in 0 blocks      # Accessible but not freed
   suppressed: 0 bytes in 0 blocks           # Ignored by suppressions
```

### Common Valgrind Errors

```
Invalid read of size 4          # Reading beyond allocation
Invalid write of size 8         # Writing beyond allocation
Invalid free()                  # Freeing invalid pointer
Mismatched free() / delete      # Using wrong deallocator
Use of uninitialised value      # Using uninitialized memory
Conditional jump or move        # Branch on uninitialized value
```

---

## Compiler Flags Quick Reference

### Debug Flags

```cmake
-g                      # Include debug symbols
-O0                     # No optimization (easier debugging)
-Wall                   # Enable all warnings
-Wextra                 # Extra warnings
-Wpedantic              # Strict standard compliance
-fsanitize=address      # AddressSanitizer (memory errors)
-fsanitize=undefined    # UndefinedBehaviorSanitizer
```

### Release Flags

```cmake
-O2                     # Standard optimization
-O3                     # Maximum optimization
-Os                     # Optimize for size
-DNDEBUG                # Disable assert() macro
-march=native           # Optimize for CPU
-flto                   # Link-time optimization
```

### Warning Flags

```cmake
-Wall                   # Common warnings
-Wextra                 # Extra warnings
-Werror                 # Treat warnings as errors
-Wconversion            # Implicit conversions
-Wshadow                # Variable shadowing
-Wnon-virtual-dtor      # Non-virtual destructor
```

---

## VS Code Keyboard Shortcuts

### Debugging

| Shortcut | Action |
|----------|--------|
| `F5` | Start/Continue debugging |
| `Shift+F5` | Stop debugging |
| `Ctrl+Shift+F5` | Restart debugging |
| `F9` | Toggle breakpoint |
| `F10` | Step over |
| `F11` | Step into |
| `Shift+F11` | Step out |

### Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl+Space` | Trigger IntelliSense |
| `Ctrl+.` | Quick fix |
| `F12` | Go to definition |
| `Shift+F12` | Find references |
| `Alt+Shift+F` | Format document |
| `Ctrl+/` | Toggle comment |
| `Ctrl+Shift+B` | Build (run task) |

### Navigation

| Shortcut | Action |
|----------|--------|
| `Ctrl+P` | Quick open file |
| `Ctrl+Shift+O` | Go to symbol |
| `Ctrl+T` | Go to symbol in workspace |
| `Alt+Left/Right` | Navigate backward/forward |
| `Ctrl+G` | Go to line |

---

## Matrix Class Quick Reference

### Creating Matrices

```cpp
// Constructor
Matrix m1(2, 3);         // 2x3 matrix, initialized to 0
Matrix m2(3, 3);         // 3x3 matrix

// Copy constructor
Matrix m3 = m1;          // Deep copy
Matrix m4(m1);           // Deep copy
```

### Setting/Getting Values

```cpp
// Set individual element
m1.set(0, 0, 5.0);       // Set row 0, col 0 to 5.0
m1.set(1, 2, 3.14);      // Set row 1, col 2 to 3.14

// Get individual element
double val = m1.get(0, 0);  // Get row 0, col 0

// Set all values
m1.fill(1.0);            // Fill with 1.0
```

### Operations

```cpp
// Addition
Matrix sum = m1.add(m2);         // m1 + m2

// Subtraction
Matrix diff = m1.subtract(m2);   // m1 - m2

// Multiplication
Matrix prod = m1.multiply(m2);   // m1 * m2 (matrix multiplication)

// Element-wise multiplication
Matrix elem = m1.elementWise(m2); // Element-wise product

// Scalar operations
Matrix scaled = m1.scale(2.0);   // Multiply all elements by 2.0

// Transpose
Matrix trans = m1.transpose();   // Transpose matrix
```

### Display

```cpp
// Print to console
m1.print();              // Display matrix with formatting
```

### Memory Management

```cpp
// Stack allocation (automatic cleanup)
{
    Matrix m(10, 10);
    // ... use matrix ...
}  // Automatically freed

// Heap allocation (manual cleanup)
Matrix* m = new Matrix(10, 10);
// ... use matrix ...
delete m;  // Must free manually

// Smart pointer (recommended)
auto m = std::make_unique<Matrix>(10, 10);
// ... use matrix ...
// Automatically freed
```

---

## Common Error Messages & Solutions

### Compilation Errors

**Error:** `fatal error: nn/matrix.h: No such file or directory`
```
Solution: Check include_directories() in CMakeLists.txt
Verify: include_directories(${PROJECT_SOURCE_DIR}/include)
```

**Error:** `undefined reference to 'Matrix::Matrix(int, int)'`
```
Solution: Link the library
Fix: target_link_libraries(matrix_example nn_lib)
```

**Error:** `error: 'Matrix' does not name a type`
```
Solution: Missing include or forward declaration
Fix: #include "nn/matrix.h"
```

### Linker Errors

**Error:** `multiple definition of 'Matrix::add'`
```
Solution: Function defined in header (should be in .cpp)
Fix: Move implementation to matrix.cpp
```

**Error:** `undefined reference to '__asan_...'`
```
Solution: Missing sanitizer library
Fix: Add -fsanitize=address to linker flags too
```

### Runtime Errors

**Error:** `Segmentation fault`
```
Common causes:
- Accessing invalid memory (nullptr, freed pointer)
- Array out of bounds
- Stack overflow

Debug: gdb ./program
      run
      backtrace
```

**Error:** `*** Error in './program': double free or corruption`
```
Cause: Deleting same memory twice
Debug: valgrind ./program
```

**Error:** `Memory leak detected`
```
Cause: new without delete
Debug: valgrind --leak-check=full ./program
```

---

## Performance Tips

### Compilation

```bash
# Fast compilation (Debug)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Optimized compilation (Release)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Parallel compilation
make -j$(nproc)           # Use all CPU cores
make -j8                  # Use 8 cores
```

### Memory

```cpp
// Reserve capacity
std::vector<Matrix> vec;
vec.reserve(1000);        // Avoid reallocations

// Pass by reference
void process(const Matrix& m);  // No copy

// Use move semantics
Matrix result = createMatrix(); // Move, not copy

// Smart pointers
auto m = std::make_unique<Matrix>(10, 10);  // Modern C++
```

### Code Optimization

```cpp
// Enable compiler optimizations
-O2          // Good balance
-O3          // Maximum optimization
-march=native // Use CPU-specific instructions

// Profile code
gprof ./program          // GNU profiler
perf record ./program    // Linux perf tool
perf report
```

---

## Useful Commands Summary

```bash
# Build
cmake .. && make

# Build and run
cmake .. && make && ./matrix_example

# Clean and rebuild
rm -rf build && mkdir build && cd build && cmake .. && make

# Debug
gdb ./matrix_example

# Memory check
valgrind --leak-check=full ./matrix_example

# Profile
time ./matrix_example

# Check for memory errors (fast)
# Add to CMakeLists.txt: -fsanitize=address
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address" .. && make

# Run with output
./matrix_example 2>&1 | tee output.txt
```

---

## Project File Structure

```
project/
├── CMakeLists.txt          # Build configuration
├── build.sh                # Build script
├── .gitignore              # Git ignore file
│
├── include/nn/             # Public headers
│   └── matrix.h
│
├── src/                    # Implementation
│   └── matrix.cpp
│
├── example/                # Example programs
│   └── matrix_example.cpp
│
├── docs/                   # Documentation
│   ├── CMAKE_COMPLETE_GUIDE.md
│   ├── MEMORY_MANAGEMENT_GUIDE.md
│   └── QUICK_REFERENCE.md
│
├── .vscode/                # VS Code configuration
│   ├── launch.json         # Debug configuration
│   ├── tasks.json          # Build tasks
│   └── settings.json       # Editor settings
│
└── build/                  # Build output (gitignored)
    ├── matrix_example      # Executable
    └── libnn_lib.a         # Library
```

---

This quick reference covers the most common tasks and commands for C++ neural network development!

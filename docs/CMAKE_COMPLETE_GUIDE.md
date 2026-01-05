# Complete CMake Guide for C++ Neural Network Project

## Table of Contents
1. [CMake Basics](#cmake-basics)
2. [CMakeLists.txt Explained Line by Line](#cmakelists-explained)
3. [Compilation Process](#compilation-process)
4. [Include Files and Folder Linking](#include-files-linking)
5. [How Executable is Created](#executable-creation)
6. [Debugging Process](#debugging-process)
7. [Project Structure Best Practices](#project-structure)

---

## 1. CMake Basics {#cmake-basics}

### What is CMake?
CMake is a **cross-platform build system generator**. It doesn't compile your code directly; instead, it generates build files (like Makefiles on Linux, Visual Studio projects on Windows) that are then used to compile your code.

### Why Use CMake?
- **Cross-platform**: Same CMakeLists.txt works on Windows, Linux, macOS
- **Dependency management**: Automatically tracks which files need recompilation
- **Scalability**: Easy to manage large projects
- **IDE integration**: Works with VS Code, Visual Studio, CLion, etc.

### CMake Workflow
```
CMakeLists.txt → CMake → Build System (Makefile) → Compiler → Executable
```

---

## 2. CMakeLists.txt Explained Line by Line {#cmakelists-explained}

Let's break down our project's CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.10)
```
**Purpose**: Specifies minimum CMake version required
- Ensures compatibility with CMake features used
- 3.10 is a common baseline (released 2017)
- If user has older version, CMake will show error

```cmake
project(NeuralNetworkFromScratch VERSION 1.0)
```
**Purpose**: Defines the project
- **project()**: Creates a project named "NeuralNetworkFromScratch"
- **VERSION 1.0**: Sets project version (optional but good practice)
- Creates variables: `PROJECT_NAME`, `PROJECT_VERSION`

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
```
**Purpose**: Sets C++ language standard
- **CMAKE_CXX_STANDARD 17**: Use C++17 features
- **CMAKE_CXX_STANDARD_REQUIRED True**: Fail if compiler doesn't support C++17
- Enables modern C++ features (auto, lambdas, std::optional, etc.)

```cmake
set(CMAKE_BUILD_TYPE Debug)
```
**Purpose**: Sets build configuration
- **Debug**: Includes debugging symbols, no optimization
- **Release**: Optimizations enabled, no debug symbols
- **RelWithDebInfo**: Release with debug info
- **MinSizeRel**: Optimized for size

```cmake
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
```
**Purpose**: Compiler flags for Debug mode
- **-g**: Include debugging symbols (for gdb)
- **-O0**: No optimization (easier debugging)
- **-Wall**: Enable all common warnings
- **-Wextra**: Enable extra warnings

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
```
**Purpose**: Compiler flags for Release mode
- **-O3**: Maximum optimization
- **-DNDEBUG**: Define NDEBUG macro (disables assert())

```cmake
include_directories(${PROJECT_SOURCE_DIR}/include)
```
**Purpose**: Add include directories
- **include_directories()**: Tells compiler where to find header files
- **${PROJECT_SOURCE_DIR}**: Variable containing project root path
- Makes `#include "nn/matrix.h"` work from anywhere

```cmake
file(GLOB_RECURSE SOURCES "src/*.cpp")
```
**Purpose**: Collect all source files
- **file(GLOB_RECURSE ...)**: Find files matching pattern
- **SOURCES**: Variable name to store file list
- **"src/*.cpp"**: Pattern to match all .cpp files in src/
- **RECURSE**: Search subdirectories too

```cmake
add_library(nn_lib STATIC ${SOURCES})
```
**Purpose**: Create a library from source files
- **add_library()**: Creates a library target
- **nn_lib**: Name of the library
- **STATIC**: Creates static library (.a on Linux, .lib on Windows)
- **${SOURCES}**: Source files to compile into library
- Alternative: **SHARED** for dynamic library (.so, .dll)

```cmake
add_executable(matrix_example example/matrix_example.cpp)
```
**Purpose**: Create an executable
- **add_executable()**: Creates executable target
- **matrix_example**: Name of executable
- **example/matrix_example.cpp**: Source file with main()

```cmake
target_link_libraries(matrix_example nn_lib)
```
**Purpose**: Link library to executable
- **target_link_libraries()**: Links libraries to target
- **matrix_example**: Target that needs the library
- **nn_lib**: Library to link
- This connects the executable to the library code

---

## 3. Compilation Process {#compilation-process}

### Complete Compilation Pipeline

```
Source Code → Preprocessor → Compiler → Assembler → Linker → Executable
```

#### Step 1: Preprocessing
```bash
g++ -E src/matrix.cpp -I include/ > matrix.i
```
- Handles `#include`, `#define`, `#ifdef`
- Expands macros
- Removes comments
- Output: `.i` file (preprocessed source)

**What happens:**
- `#include "nn/matrix.h"` is replaced with actual header content
- `#define` macros are substituted
- Conditional compilation (`#ifdef`) is resolved

#### Step 2: Compilation
```bash
g++ -S matrix.i -o matrix.s
```
- Converts C++ to assembly language
- Performs syntax checking
- Applies optimizations
- Output: `.s` file (assembly code)

**What happens:**
- C++ code converted to CPU instructions
- Optimizations applied (if -O flag used)
- Type checking performed

#### Step 3: Assembly
```bash
as matrix.s -o matrix.o
```
- Converts assembly to machine code
- Creates object file
- Output: `.o` file (object file)

**What happens:**
- Assembly instructions → binary machine code
- Creates relocatable object file
- Symbol table created (functions, variables)

#### Step 4: Linking
```bash
g++ matrix.o example.o -o matrix_example
```
- Combines object files
- Resolves references between files
- Links with libraries
- Output: Executable file

**What happens:**
- Function calls are connected to implementations
- Memory addresses are finalized
- Libraries are included
- Final executable is created

### CMake Simplifies This

Instead of running all these commands manually, CMake does:

```bash
mkdir build
cd build
cmake ..        # Generate build files
make            # Run compilation pipeline
```

CMake automatically:
- Tracks dependencies
- Only recompiles changed files
- Handles include paths
- Links libraries correctly
- Works across platforms

---

## 4. Include Files and Folder Linking {#include-files-linking}

### How Include Directories Work

#### In CMakeLists.txt:
```cmake
include_directories(${PROJECT_SOURCE_DIR}/include)
```

This tells the compiler: "When you see `#include "something"`, look in the `include/` folder."

#### In Source Files:
```cpp
#include "nn/matrix.h"  // Compiler looks in: include/nn/matrix.h
```

### Include Search Process

When compiler sees `#include "nn/matrix.h"`:

1. **Check current directory**
2. **Check include_directories()** paths:
   - `/media/.../NNFROMSCRATCH/include/`
3. **Check system directories**:
   - `/usr/include/`
   - `/usr/local/include/`

### Project Structure and Includes

```
project/
├── include/           # Public headers
│   └── nn/
│       └── matrix.h   # #include "nn/matrix.h"
├── src/               # Implementation files
│   └── matrix.cpp     # #include "nn/matrix.h"
└── example/           # Example programs
    └── matrix_example.cpp  # #include "nn/matrix.h"
```

**Why this structure?**
- **Separation**: Interface (header) separate from implementation (cpp)
- **Organization**: Related files in subdirectories
- **Clarity**: `#include "nn/matrix.h"` shows it's from nn namespace
- **Scalability**: Easy to add more components

### Header vs Source Files

**Header Files (.h)**
- Declarations (what exists)
- Class definitions
- Function prototypes
- Inline functions
- Should have include guards

**Source Files (.cpp)**
- Implementations (how it works)
- Function bodies
- Static variables
- Only compiled once

---

## 5. How Executable is Created {#executable-creation}

### From Source to Executable: Complete Flow

```
Step 1: Compile Library
  src/matrix.cpp → matrix.o (object file)
  ↓
  ar rcs libnn_lib.a matrix.o (static library)

Step 2: Compile Example
  example/matrix_example.cpp → matrix_example.o

Step 3: Link Everything
  matrix_example.o + libnn_lib.a → matrix_example (executable)
```

### Detailed Linking Process

#### 1. Symbol Resolution
```cpp
// In matrix_example.cpp
Matrix m1(2, 2);  // Compiler: "Where is Matrix constructor?"
```

Linker searches:
- `matrix_example.o` - Not found
- `libnn_lib.a` - Found! Links to Matrix::Matrix()

#### 2. Address Binding
- Object files have relative addresses
- Linker assigns final memory addresses
- Updates all references

#### 3. Library Integration
**Static Library (our case):**
- Code copied into executable
- Larger executable size
- No external dependencies
- Faster startup

**Dynamic Library (alternative):**
- Code stays in .so/.dll file
- Smaller executable
- Shared between programs
- Requires library at runtime

### CMake Build Process

```bash
cmake ..     # Configuration phase
```
**CMake does:**
1. Reads CMakeLists.txt
2. Finds compiler (g++, clang++, msvc)
3. Checks C++17 support
4. Generates Makefile (or ninja.build, etc.)
5. Creates build rules

```bash
make         # Build phase
```
**Make does:**
1. Checks what changed
2. Compiles changed .cpp files
3. Creates object files (.o)
4. Builds library (libnn_lib.a)
5. Compiles example program
6. Links everything
7. Creates executable

### Build Artifacts

After `make`, you get:
```
build/
├── CMakeFiles/              # CMake internal files
├── libnn_lib.a              # Static library
├── matrix_example           # Executable
└── Makefile                 # Generated build rules
```

---

## 6. Debugging Process {#debugging-process}

### Enabling Debug Mode

Already configured in CMakeLists.txt:
```cmake
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
```

### Using GDB (GNU Debugger)

#### Starting GDB
```bash
cd build
gdb ./matrix_example
```

#### Essential GDB Commands

**1. Running the Program**
```gdb
(gdb) run                    # Run program
(gdb) run arg1 arg2          # Run with arguments
```

**2. Setting Breakpoints**
```gdb
(gdb) break main             # Break at main function
(gdb) break matrix.cpp:45    # Break at line 45 in matrix.cpp
(gdb) break Matrix::add      # Break at Matrix::add method
```

**3. Stepping Through Code**
```gdb
(gdb) step                   # Step into functions
(gdb) next                   # Step over functions
(gdb) continue               # Continue until next breakpoint
(gdb) finish                 # Finish current function
```

**4. Inspecting Variables**
```gdb
(gdb) print m1               # Print variable m1
(gdb) print m1.rows          # Print member variable
(gdb) print *this            # Print current object
(gdb) display m1             # Auto-display m1 at each step
```

**5. Examining Memory**
```gdb
(gdb) info locals            # Show local variables
(gdb) info args              # Show function arguments
(gdb) backtrace              # Show call stack
(gdb) frame 2                # Switch to frame 2 in stack
```

**6. Watchpoints**
```gdb
(gdb) watch m1.data          # Break when data changes
(gdb) watch *(data + 5)      # Break when memory location changes
```

### Debugging Example Session

```bash
$ gdb ./matrix_example

(gdb) break Matrix::add
Breakpoint 1 at matrix.cpp:50

(gdb) run
Starting program: ./matrix_example

Breakpoint 1, Matrix::add (this=0x7fffffffe000, other=...) at matrix.cpp:50

(gdb) print this->rows
$1 = 2

(gdb) print this->cols
$2 = 2

(gdb) print other.data[0]
$3 = 5

(gdb) step              # Step into function
(gdb) next              # Execute next line
(gdb) continue          # Continue execution
```

### Debugging Memory Issues

#### Using Valgrind (Memory Leak Detector)
```bash
# Install valgrind
sudo apt-get install valgrind

# Run with valgrind
valgrind --leak-check=full ./matrix_example
```

**What Valgrind detects:**
- Memory leaks
- Use of uninitialized memory
- Invalid memory access
- Double free errors

#### Using AddressSanitizer (Faster Alternative)

Modify CMakeLists.txt:
```cmake
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -fsanitize=address")
```

Rebuild and run:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
./matrix_example
```

**AddressSanitizer detects:**
- Heap buffer overflow
- Stack buffer overflow
- Use after free
- Memory leaks (faster than Valgrind)

### VS Code Debugging Setup

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Matrix Example",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/matrix_example",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}/build",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "build",
            "miDebuggerPath": "/usr/bin/gdb"
        }
    ]
}
```

Create `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build",
            "type": "shell",
            "command": "cd build && cmake .. && make",
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```

**Now you can:**
- Set breakpoints by clicking line numbers
- Press F5 to start debugging
- Use debug toolbar (step over, step into, continue)
- Hover over variables to see values
- View call stack in side panel

---

## 7. Project Structure Best Practices {#project-structure}

### Recommended Structure

```
NeuralNetworkFromScratch/
├── CMakeLists.txt           # Main build configuration
├── README.md                # Project documentation
├── .gitignore               # Git ignore rules
│
├── include/                 # Public headers
│   └── nn/
│       ├── matrix.h
│       ├── layer.h
│       ├── activation.h
│       └── network.h
│
├── src/                     # Implementation files
│   ├── matrix.cpp
│   ├── layer.cpp
│   ├── activation.cpp
│   └── network.cpp
│
├── example/                 # Example programs
│   ├── matrix_example.cpp
│   ├── simple_network.cpp
│   └── mnist_training.cpp
│
├── test/                    # Unit tests
│   ├── test_matrix.cpp
│   └── test_layer.cpp
│
├── docs/                    # Documentation
│   ├── CMAKE_GUIDE.md
│   └── API_REFERENCE.md
│
├── build/                   # Build directory (gitignored)
│   └── (generated files)
│
└── external/                # Third-party libraries
    └── googletest/
```

### .gitignore for C++ Projects

Create `.gitignore`:
```
# Build directories
build/
cmake-build-*/

# Compiled files
*.o
*.a
*.so
*.exe

# IDE files
.vscode/
.idea/
*.swp

# CMake cache
CMakeCache.txt
CMakeFiles/
```

### Multi-Directory CMake Setup

For larger projects, use subdirectory CMakeLists:

**Root CMakeLists.txt:**
```cmake
cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)
```

**src/CMakeLists.txt:**
```cmake
file(GLOB_RECURSE SOURCES "*.cpp")
add_library(nn_lib STATIC ${SOURCES})
target_include_directories(nn_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
```

**example/CMakeLists.txt:**
```cmake
add_executable(matrix_example matrix_example.cpp)
target_link_libraries(matrix_example nn_lib)
```

---

## Common CMake Commands Reference

### Configuration
```bash
cmake ..                          # Basic configuration
cmake .. -DCMAKE_BUILD_TYPE=Debug # Debug build
cmake .. -DCMAKE_BUILD_TYPE=Release # Release build
cmake .. -G Ninja                 # Use Ninja instead of Make
```

### Building
```bash
make                              # Build all targets
make matrix_example               # Build specific target
make -j8                          # Parallel build (8 jobs)
cmake --build .                   # Cross-platform build command
```

### Cleaning
```bash
make clean                        # Remove built files
rm -rf build/*                    # Clean everything
```

### Installation
```bash
make install                      # Install to system
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
```

---

## Troubleshooting

### Common Issues

**1. "No such file or directory" when including header**
- Check `include_directories()` path
- Verify file exists: `ls include/nn/matrix.h`
- Check include statement: `#include "nn/matrix.h"`

**2. "Undefined reference" linking error**
- Add library to `target_link_libraries()`
- Check function is implemented in .cpp file
- Verify library is built before executable

**3. CMake can't find compiler**
```bash
export CXX=/usr/bin/g++
cmake ..
```

**4. Cache problems**
```bash
rm -rf build/*
cmake ..
make
```

---

## Summary: Complete Workflow

### 1. Write Code
```cpp
// include/nn/matrix.h - Declaration
// src/matrix.cpp - Implementation  
// example/matrix_example.cpp - Usage
```

### 2. Configure CMake
```cmake
# CMakeLists.txt
# Define project, sources, includes, targets
```

### 3. Build
```bash
mkdir build && cd build
cmake ..              # Generate build system
make                  # Compile everything
```

### 4. Run
```bash
./matrix_example      # Execute
```

### 5. Debug (if needed)
```bash
gdb ./matrix_example  # Interactive debugging
valgrind ./matrix_example  # Memory checking
```

### 6. Iterate
- Modify code
- `make` (only rebuilds changed files)
- Run/test

---

## Next Steps for Learning

1. **Practice**: Modify matrix.cpp, add new operations
2. **Expand**: Add Layer class, Activation functions
3. **Test**: Write unit tests with Google Test
4. **Optimize**: Profile code, add SIMD operations
5. **Deploy**: Create installable package with CPack

This guide covers everything you need to understand CMake, compilation, and debugging for C++ neural network development!

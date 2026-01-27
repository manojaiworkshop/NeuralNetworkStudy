#!/usr/bin/env python3
"""
C++ LibTorch inference wrapper - Run with proper library paths
"""

import subprocess
import sys
import os

# Get the directory containing torch
import torch
torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')

# Get CUDA library directory
cuda_lib_dir = "/usr/lib/x86_64-linux-gnu"

# Set environment
env = os.environ.copy()
if 'LD_LIBRARY_PATH' in env:
    env['LD_LIBRARY_PATH'] = f"{torch_lib_dir}:{cuda_lib_dir}:{env['LD_LIBRARY_PATH']}"
else:
    env['LD_LIBRARY_PATH'] = f"{torch_lib_dir}:{cuda_lib_dir}"

# Run the C++ executable
executable = "./build/joint_nlu_inference"
if not os.path.exists(executable):
    print(f"❌ Executable not found: {executable}")
    print("Please build first: cd build && make joint_nlu_inference")
    sys.exit(1)

# Execute
try:
    subprocess.run([executable], env=env, check=True)
except KeyboardInterrupt:
    print("\n👋 Goodbye!")
except subprocess.CalledProcessError as e:
    print(f"❌ Error running inference: {e}")
    sys.exit(1)

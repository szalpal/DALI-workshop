# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import subprocess
import time
from collections import deque
import numpy as np

def get_gpu_utilization():
    """Get GPU utilization using nvidia-smi"""
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'])
        return [int(x) for x in output.decode('utf-8').strip().split('\n')]
    except:
        return [0]  # Return 0 if nvidia-smi fails

def draw_histogram(data, width=100, height=20):
    """Draw a histogram in the terminal"""
    # Calculate histogram
    hist, bins = np.histogram(data, bins=width, range=(0, 100))
    
    # Find maximum count for scaling
    max_count = max(hist) if len(hist) > 0 else 1
    
    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")
    
    # Draw histogram
    for i in range(height, 0, -1):
        line = ""
        for count in hist:
            if count >= (i * max_count / height):
                line += "█"
            else:
                line += " "
        print(line)
    
    # Draw x-axis
    print("0%" + " " * (width-4) + "100%")
    
    # Show current utilization
    current_util = data[-1] if data else 0
    print(f"\nCurrent GPU Utilization: {current_util}%")

def main():
    # Initialize data storage
    data = deque(maxlen=1000)  # Store last 1000 measurements
    
    try:
        while True:
            # Get GPU utilization
            utilizations = get_gpu_utilization()
            for util in utilizations:
                data.append(util)
            
            # Draw histogram
            draw_histogram(list(data))
            
            time.sleep(0.3)  # Update every 0.3 seconds
            
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()

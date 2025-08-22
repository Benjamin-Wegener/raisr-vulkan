#!/usr/bin/env python3
"""
Script to convert trained RAISR filters to binary format for C implementation.

This script converts the trained filter from Python pickle format to a binary
format that can be read by the C implementation. The C program expects a binary
file with the following format:
- First 4 bytes: unsigned int representing the number of filter elements
- Remaining bytes: double precision floating point values of the filter

Usage:
    python save_filter.py
    python save_filter.py -i filter_BSDS500 -o filter.bin
"""

import pickle
import numpy as np
import struct
import argparse
import os

def save_filter(input_file, output_file):
    """Convert trained filter from pickle to binary format."""
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return False
    
    try:
        # Load the trained filter
        h = pickle.load(open(input_file, 'rb'))
        print(f'Filter shape: {h.shape}')
        print(f'Filter dtype: {h.dtype}')

        # Flatten the filter
        h_flat = h.flatten()
        print(f'Flattened filter shape: {h_flat.shape}')

        # Save as binary file
        with open(output_file, 'wb') as f:
            # Write the number of elements first
            f.write(struct.pack('I', len(h_flat)))
            # Write the filter data as float64
            np.array(h_flat, dtype=np.float64).tofile(f)

        print(f'Filter successfully saved as binary file with {len(h_flat)} elements')
        print(f'Output file: {output_file}')
        return True
    except Exception as e:
        print(f"Error saving filter: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert trained RAISR filters to binary format for C implementation')
    parser.add_argument('-i', '--input', default='filter_BSDS500', 
                        help='Input pickle file (default: filter_BSDS500)')
    parser.add_argument('-o', '--output', default='filter.bin',
                        help='Output binary file (default: filter.bin)')
    
    args = parser.parse_args()
    
    print(f"Converting filter from '{args.input}' to '{args.output}'")
    success = save_filter(args.input, args.output)
    
    if success:
        print("Filter conversion completed successfully!")
    else:
        print("Filter conversion failed!")

if __name__ == '__main__':
    main()

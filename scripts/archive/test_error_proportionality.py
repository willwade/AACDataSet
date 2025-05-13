#!/usr/bin/env python3
"""
Test script to verify that the number of errors is proportional to the error rate.
This script tests the generate_noisy_utterance function with different error rates
and measures the edit distance between the original and noisy utterances.
"""

import sys
from pathlib import Path
lib_path = str((Path(__file__).parent.parent / "lib").resolve())
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.augment_aac_data import generate_noisy_utterance, create_qwerty_grid

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def test_error_proportionality():
    """Test that the number of errors is proportional to the error rate."""
    # Create a grid for testing
    grid = create_qwerty_grid(3, 10)
    
    # Test cases of different lengths
    test_cases = [
        "Hello",
        "This is a test",
        "The quick brown fox jumps over the lazy dog",
        "A longer sentence to test the proportionality of errors introduced based on the error rate and text length."
    ]
    
    # Test with different error rates
    error_rates = [0.05, 0.15, 0.25, 0.35, 0.5]
    
    print("Text Length | Error Rate | Edit Distance | % Change")
    print("-" * 60)
    
    for text in test_cases:
        for error_rate in error_rates:
            # Generate a noisy utterance
            noisy = generate_noisy_utterance(text, grid, error_rate=error_rate)
            
            # Calculate the edit distance
            distance = levenshtein_distance(text, noisy)
            
            # Calculate the percentage of characters changed
            percent_change = (distance / len(text)) * 100
            
            print(f"{len(text):11} | {error_rate:10.2f} | {distance:12} | {percent_change:7.2f}%")
        
        # Add a separator between test cases
        print("-" * 60)

if __name__ == "__main__":
    test_error_proportionality()

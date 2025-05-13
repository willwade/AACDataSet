#!/usr/bin/env python3
"""
Test script for the generate_noisy_utterance function.
This script tests that the function always introduces at least one error.
"""

import sys
from pathlib import Path

lib_path = str((Path(__file__).parent.parent / "lib").resolve())
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.augment_aac_data import generate_noisy_utterance, create_qwerty_grid


def test_noisy_utterance():
    """Test that generate_noisy_utterance always introduces at least one error."""
    # Create a grid for testing
    grid = create_qwerty_grid(3, 10)

    # Test cases
    test_cases = [
        "Hello",
        "A",
        "Hi",
        "Test",
        "This is a longer sentence to test.",
        "Short",
        "OK",
        "Yes",
        "No",
        "Maybe",
    ]

    # Test with different error rates
    error_rates = [0.05, 0.15, 0.25, 0.35]

    for text in test_cases:
        print(f"\nTesting with text: '{text}'")
        for error_rate in error_rates:
            # Generate a noisy utterance
            noisy = generate_noisy_utterance(text, grid, error_rate=error_rate)

            # Check if the noisy utterance is different from the original
            is_different = noisy != text

            print(
                f"  Error rate {error_rate:.2f}: '{noisy}' - {'DIFFERENT' if is_different else 'SAME'}"
            )

            # Assert that the noisy utterance is different from the original
            if not is_different:
                print(
                    f"  ERROR: Noisy utterance is the same as the original with error rate {error_rate}"
                )


if __name__ == "__main__":
    test_noisy_utterance()

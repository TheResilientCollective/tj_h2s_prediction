#!/usr/bin/env python3
"""
Batch H2S Prediction — DEPRECATED
===================================

This script is deprecated. Use predict_h2s.py directly, which now handles
all stations by default:

    python predict_h2s.py --input data.parquet --models ./models --output ./output

For backwards compatibility, this script translates the old --obs argument
and delegates to predict_h2s.main().
"""

import sys

from predict_h2s import main as predict_main


def main():
    # Translate --obs to --input for backwards compatibility
    args = sys.argv[1:]
    translated = []
    i = 0
    while i < len(args):
        if args[i] == '--obs':
            translated.append('--input')
        else:
            translated.append(args[i])
        i += 1

    sys.argv = [sys.argv[0]] + translated
    print("NOTE: batch_predict.py is deprecated. Use predict_h2s.py --input instead.\n")
    return predict_main()


if __name__ == '__main__':
    exit(main())
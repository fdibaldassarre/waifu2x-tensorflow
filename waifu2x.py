#!/usr/bin/env python3

import argparse
import os
import sys

from src.Waifu2x import scale
from src.Waifu2x import denoise
from src.Waifu2x import denoise_scale

OPERATION_NOISE = "noise"
OPERATION_SCALE = "scale"
OPERATION_NOISE_SCALE = "noise_scale"

VALID_OPERATIONS = [OPERATION_NOISE, OPERATION_SCALE, OPERATION_NOISE_SCALE]

parser = argparse.ArgumentParser(description="Waifu2x")
parser.add_argument("-m", dest="operation", help="Operation: noise, scale, noise_scale")
parser.add_argument("-i", dest="input", help="Input image")
parser.add_argument("-o", dest="output", help="Output image")
parser.add_argument("-noise_level", dest="noise", type=int, default=0, help="Noise level from 0 to 3")

args = parser.parse_args()

operation = args.operation
input_path = args.input
output_path = args.output
noise_level = args.noise

if input_path is None:
    print("Input image not specified")
    sys.exit(1)

if not os.path.exists(input_path):
    print("Input image %s does not exists" % input_path)
    sys.exit(1)

if output_path is None:
    print("Please specify the output image")
    sys.exit(2)

if operation is None:
    print("Please specify the operation: noise, scale or noise_scale.")
    sys.exit(3)

noise_level = min(max(0, noise_level), 3)

if operation == OPERATION_NOISE:
    denoise(input_path, output_path, noise_level)
elif operation == OPERATION_SCALE:
    if noise_level != 0:
        print("Warning! Noise level is ignored for scale operation. Use noise_scale to scale and remove noise.")
    scale(input_path, output_path)
elif operation == OPERATION_NOISE_SCALE:
    denoise_scale(input_path, output_path, noise_level)
else:
    print("Invalid operation %s. Valid operations are noise, scale, noise_scale." % operation)
    sys.exit(4)

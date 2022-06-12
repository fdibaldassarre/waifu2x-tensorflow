#!/usr/bin/env python3

import argparse
import os
import sys

from src.Waifu2x import scale
from src.Waifu2x import denoise
from src.Waifu2x import denoise_scale
from src.Waifu2x import OP_SCALE, OP_NOISE, OP_NOISE_SCALE
from src.Waifu2x import Waifu2x

from src.Utils import get_images_from_file
from src.Utils import fill_output_file_placeholders

VALID_OPERATIONS = {OP_SCALE, OP_NOISE, OP_NOISE_SCALE}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waifu2x")
    parser.add_argument("-m", dest="operation", help="Operation: noise, scale, noise_scale")
    parser.add_argument("-i", dest="input", help="Input image")
    parser.add_argument("-o", dest="output", help="Output image")
    parser.add_argument("-noise_level", dest="noise", type=int, default=0, help="Noise level from 0 to 3")
    parser.add_argument("-l", dest="image_list", help="List of images, one path per line")

    args = parser.parse_args()

    operation = args.operation
    input_path = args.input
    output_path = args.output
    noise_level = args.noise
    image_list = args.image_list

    if operation is None:
        print("Please specify the operation: noise, scale or noise_scale.")
        sys.exit(3)

    if operation not in VALID_OPERATIONS:
        print("Invalid operation " + operation + ". Valid operations are: noise, scale or noise_scale.")
        sys.exit(4)

    noise_level = min(max(0, noise_level), 3)

    if image_list is not None and os.path.exists(image_list):
        batch_mode = True
    else:
        batch_mode = False

    if not batch_mode:
        # Single file validation
        if input_path is None:
            print("Input image not specified")
            sys.exit(1)

        if not os.path.exists(input_path):
            print("Input image %s does not exists" % input_path)
            sys.exit(1)

    # Create Waifu2x
    if operation == OP_SCALE:
        if noise_level != 0:
            print("Warning! Noise level is ignored for scale operation. Use noise_scale to scale and remove noise.")
        waifu2x = Waifu2x(OP_SCALE)
    else:
        waifu2x = Waifu2x(operation, noise_level)

    if batch_mode:
        input_files = get_images_from_file(image_list)
    else:
        input_files = [input_path]

    for i, input_file in enumerate(input_files):
        output_file = fill_output_file_placeholders(output_path, i, input_file)
        print("Converting {input_path} to {output_path}".format(input_path=os.path.abspath(input_file),
                                                                output_path=os.path.abspath(output_file)))
        waifu2x.run(input_file, output_file)


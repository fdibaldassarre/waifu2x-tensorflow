#!/usr/bin/env python3

import os


def get_images_from_file(path):
    with open(path, "r") as hand:
        for line in hand:
            line = line.strip()
            if not os.path.exists(line):
                print("Skipping missing file: " + line)
            else:
                yield line


def fill_output_file_placeholders(output_path, input_idx, input_file):
    if "%s" in output_path:
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_path = output_path.replace("%s", input_name)
    if "%d" in output_path:
        output_path = output_path % (input_idx,)
    return output_path

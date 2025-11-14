# waifu2x-tensorflow
Implementation of nagadomi [Waifu2x](https://github.com/nagadomi/waifu2x) in Tensorflow.

## Requirements

- Python 3.13
- Tensorflow 2.20
- Pillow 12
- numpy 2.3.4

```sh
uv venv
uv sync
```

## Usage

Command line options are the same as the original waifu2x.

```sh
./waifu2x.py -i "input.png" -o "output.png" -m scale 
./waifu2x.py -i "input.png" -o "output.png" -m noise -noise_level 1
./waifu2x.py -i "input.png" -o "output.png" -m noise_scale -noise_level 1
./waifu2x.py -l "image_list.txt" -o "upscaled/%s.png" -m noise_scale -noise_level 1
```

# waifu2x-tensorflow
Implementation of nagadomi [Waifu2x](https://github.com/nagadomi/waifu2x) in Tensorflow.

## Requirements

- Python 3.10
- Tensorflow 2.9.1
- Numpy
- PIL

```sh
pip install -r requirements
```

## Usage

Command line options are the same as the original waifu2x.

```sh
./waifu2x.py -i "input.png" -o "output.png" -m scale 
./waifu2x.py -i "input.png" -o "output.png" -m noise -noise_level 1
./waifu2x.py -i "input.png" -o "output.png" -m noise_scale -noise_level 1
./waifu2x.py -l "image_list.txt" -o "upscaled/%s.png" -m noise_scale -noise_level 1
```

## Todo

- Support for photos
- Support for cunet model
- Training

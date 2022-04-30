# Video-Compression-H.261
**Video compression based on H.261 standard.**

Development environment: Python3

Install Python Library:

```shell
pip install -r requirements
```

Command to run:

```shell
python .\main__.py --src Dance.yuv --size 640x360 --fps 30 --dst Dance.avi
```

- src: file path of the source video (in the .yuv format)

- size: `height`x`width`

- fps: the frame per second of the source video

- dst: file path of the destination video (in the .avi format)

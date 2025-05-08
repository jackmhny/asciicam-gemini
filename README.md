# Color ASCII Webcam with OpenCV Face Detection

This project captures video from your webcam (/dev/video0), performs face detection using OpenCV, and renders the output as colored ASCII art in the terminal.

## Requirements

- Linux with V4L2 support (e.g., `/dev/video0`)
- V4L2 development headers (e.g., `libv4l-dev`)
- OpenCV development libraries (e.g., `libopencv-dev`)
- `pkg-config`
- Terminal with UTF-8 encoding, 24-bit true color support, and the '▀' (U+2580) character.

## Build

Use the provided Makefile:
```bash
make
```
This will produce the `ascii_cam_opencv` executable.

## Usage

```bash
./ascii_cam_opencv haarcascade_frontalface_default.xml
```

You can optionally provide a different path to the Haar Cascade XML file. Press `Ctrl+C` to exit.

## Files

- `ascii_cam_opencv.cpp`: Source code
- `haarcascade_frontalface_default.xml`: Pre-trained face detection model
- `Makefile`: Build script
- `.gitignore`: Git ignore rules
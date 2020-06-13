# Installation

1. Install [`pipenv`](https://pipenv.pypa.io/en/latest/install/#installing-pipenv)
1. Then change dir to this repo and run `pipenv install`

# Running

```
$ pipenv shell
$ python nyquist_video.py example_h8_1.toml
```

# Creating video

```
$ ffmpeg -framerate 24 -i "/tmp/nyquist_h8_1/frame_%06d.png" "/tmp/nyquist_h8_1/video.mp4"
```

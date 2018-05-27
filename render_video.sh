#!/bin/bash

ffmpeg \
  -r 30 \
  -i imgs/img_%010d.png \
  -i audio/Sierra\ Hull\ \ Black\ River\ \(OFFICIAL\ VIDEO\).wav \
  -c:v libx264 \
  -c:a aac \
  -pix_fmt yuv420p \
  -crf 23 \
  -r 30 \
  -strict -2 \
  -shortest \
  -y video-from-frames.mp4
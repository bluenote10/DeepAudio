import math
import lenientops

const SAMPLE_RATE* = 44_100

type
  SampleType* = float32

  AudioChunk* = object
    data*: seq[SampleType]

proc len*(audio: AudioChunk): int = audio.data.len

proc meanAbs*(audio: AudioChunk): float =
  var sum = 0.0
  for x in audio.data:
    sum += abs(x)
  sum / audio.data.len()
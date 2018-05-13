import math
import lenientops
import sequtils
import sugar

const SAMPLE_RATE* = 44_100

type
  SampleType* = float32

  AudioChunk* = object
    data*: seq[SampleType]

proc newAudioChunk(len: int): AudioChunk =
  AudioChunk(data: newSeq[SampleType](len))

proc toAudioChunk*(s: seq[SampleType]): AudioChunk =
  AudioChunk(data: s)

proc toAudioChunk*(s: seq[float]): AudioChunk =
  AudioChunk(data: s.map(x => x.SampleType))

proc len*(audio: AudioChunk): int = audio.data.len

proc maxAbs*(audio: AudioChunk): SampleType =
  var max = 0.SampleType
  for x in audio.data:
    if abs(x) > max:
      max = abs(x)
  max

proc mean*(audio: AudioChunk): float =
  var sum = 0.0
  for x in audio.data:
    sum += x
  sum / audio.data.len()

proc meanAbs*(audio: AudioChunk): float =
  var sum = 0.0
  for x in audio.data:
    sum += abs(x)
  sum / audio.data.len()

proc rms*(audio: AudioChunk): float =
  var sumSquared = 0.0
  for x in audio.data:
    sumSquared += x*x
  let meanSquared = sumSquared / audio.len
  return sqrt(meanSquared)


proc normalize*(audio: var AudioChunk, normalizeTo = 1.0) =
  var max = SampleType(0)
  for x in audio.data:
    if x.abs > max:
      max = x.abs
  if max > 0:
    for i in 0 ..< audio.len:
      audio.data[i] /= max

proc normalized*(audio: AudioChunk, normalizeTo = 1.0): AudioChunk =
  result = audio
  result.normalize()


proc rmsChunks*(audio: var AudioChunk, chunkSize = 32): AudioChunk =
  result = newAudioChunk(0)
  var l = 0
  var done = false
  while not done:
    var r = l + chunkSize
    if r >= audio.len:
      r = audio.len
      done = true
    let chunk = audio.data[l ..< r].toAudioChunk
    result.data.add(chunk.rms)
    l = r

import streams
import lenientops

import math
import os

type
  AudioChunk = object
    data: seq[float]

const SAMPLE_RATE = 44_100


proc generateSine*(freq: float, amp: float, duration: float): AudioChunk =
  let sampleLength = int(duration * SAMPLE_RATE)
  let period = 1.0 / freq * SAMPLE_RATE

  var data = newSeq[float](sampleLength)
  for i in 0 ..< sampleLength:
    data[i] = amp * sin(2 * PI * i / period)

  return AudioChunk(data: data)


proc convertSample*(x: float): int16 =
  int16(x * (high(int16).float + 0.5) - 0.5)


proc writeWave*(audio: AudioChunk, filename: string) =
  var s = newFileStream(filename, fmWrite)

  let numChannels = 1
  let sampleRate = 44100
  let bitsPerSample = 16
  let byteRate = sampleRate * numChannels * bitsPerSample / 8
  let blockAlign = numChannels * bitsPerSample / 8

  let audioSize = audio.data.len * numChannels * int(bitsPerSample / 8)
  let expectedFilesize = audioSize + 44

  # RIFF header
  s.write("RIFF")
  s.write(uint32(expectedFilesize - 8))
  s.write("WAVE")

  # fmt chunk
  s.write("fmt ")
  s.write(uint32(16))           # Subchunk1Size
  s.write(uint16(1))            # AudioFormat: PCM = 1
  s.write(uint16(numChannels))  # NumChannels: Mono = 1
  s.write(uint32(sampleRate))   # SampleRate
  s.write(uint32(byteRate))     # ByteRate == SampleRate * NumChannels * BitsPerSample/8
  s.write(uint16(blockAlign))   # BlockAlign == NumChannels * BitsPerSample/8
  s.write(uint16(bitsPerSample))# BitsPerSample

  let subchunkSize = audio.data.len * numChannels * bitsPerSample / 8
  s.write("data")
  s.write(uint32(subchunkSize)) # Subchunk2Size == NumSamples * NumChannels * BitsPerSample/8
  for x in audio.data:
    s.write(convertSample(x))

  s.close()
  let actualFilesize = getFileSize(filename)
  doAssert actualFilesize == expectedFilesize


if isMainModule:
  #echo system.cpuEndian
  #discard loadWave("../../crass_action_closed_hat.wav")
  doAssert convertSample(+1.0) == +32767
  doAssert convertSample(-1.0) == -32768

  let data = generateSine(440, 1.0, 1.0)
  data.writeWave("test.wav")


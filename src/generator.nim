import streams
import lenientops

import math
import os
import random
import sequtils

import arraymancer

import audiotypes

randomize(seed=42)

proc midiKeyToFreq*(midiKey: int): float =
  ## Converts a midi key to frequency
  ## Reference: https://newt.phys.unsw.edu.au/jw/notes.html
  const concertPitchKey = 69
  const concertPitchFreq = 440.0
  let keyDiff = midiKey - concertPitchKey
  return pow(2, keyDiff / 12.0) * concertPitchFreq


proc normalize*(audio: var AudioChunk, normalizeTo = 1.0) =
  var max = SampleType(0)
  for x in audio.data:
    if x.abs > max:
      max = x.abs
  if max > 0:
    for i in 0 ..< audio.len:
      audio.data[i] /= max

proc addNote*(audio: var AudioChunk, ampData: var seq[float32], midiKey: int, sampleFrom: int, sampleUpto: int, envelopeLength = 1000) =
  doAssert(sampleFrom >= 0)
  doAssert(sampleUpto < audio.len)

  let freq = midiKeyToFreq(midiKey)
  let period = 1.0 / freq * SAMPLE_RATE

  var j = 0
  for i in sampleFrom ..< sampleUpto:
    let distFromEnds = min(i - sampleFrom, sampleUpto - i)
    let amp =
      if distFromEnds >= envelopeLength:
        1.0
      else:
        (distFromEnds+1) / (envelopeLength+1)
    audio.data[i] += amp * sin(2 * PI * j / period)
    ampData[i] += amp
    j.inc


proc generateSilence*(duration: float): AudioChunk =
  let sampleLength = int(duration * SAMPLE_RATE)
  return AudioChunk(data: newSeq[SampleType](sampleLength))


proc generateSine*(freq: float, amp: float, duration: float): AudioChunk =
  let sampleLength = int(duration * SAMPLE_RATE)
  let period = 1.0 / freq * SAMPLE_RATE

  var data = newSeq[SampleType](sampleLength)
  for i in 0 ..< sampleLength:
    data[i] = amp * sin(2 * PI * i / period)

  return AudioChunk(data: data)


proc generateRandomNotes*(duration: float, numNotes: int, minNoteLength = 0.1, maxNoteLength = 1.0): tuple[audio: AudioChunk, target: Tensor[float32]] =
  var audio = generateSilence(duration)
  let maxIndex = audio.len - 1

  let minMidiKey = 21
  let maxMidiKey = 108
  var groundTruth = newSeqWith(maxMidiKey - minMidiKey + 1, newSeq[float32](audio.len))

  let minNoteLengthSamples = int(minNoteLength * SAMPLE_RATE)
  let maxNoteLengthSamples = int(maxNoteLength * SAMPLE_RATE)
  for i in 0 ..< numNotes:
    let duration = rand(minNoteLengthSamples .. maxNoteLengthSamples)
    let sampleFrom = rand(0 .. maxIndex - minNoteLengthSamples)
    let sampleUpto = min(sampleFrom + duration, audio.len - 1)
    let midiKey = rand(minMidiKey .. maxMidiKey)
    audio.addNote(groundTruth[midiKey - minMidiKey], midiKey, sampleFrom, sampleUpto)
  audio.normalize(0.5)
  return (audio: audio, target: groundTruth.toTensor)


template convertSample*(x: float): int16 =
  ## Converts a float in the range [-1.0, +1.0] to [-32768, +32767]
  doAssert(x <= +1.0)
  doAssert(x >= -1.0)
  int16(x * (high(int16).float + 0.5) - 0.5)


proc writeWave*(audio: AudioChunk, filename: string) =
  ## Simple wave writer
  ## Reference: http://soundfile.sapp.org/doc/WaveFormat/

  var s = newFileStream(filename, fmWrite)

  let numChannels = 1
  let sampleRate = 44100
  let bitsPerSample = 16
  let byteRate = sampleRate * numChannels * bitsPerSample / 8
  let blockAlign = numChannels * bitsPerSample / 8

  let audioSize = audio.len * numChannels * int(bitsPerSample / 8)
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

  let subchunkSize = audio.len * numChannels * bitsPerSample / 8
  s.write("data")
  s.write(uint32(subchunkSize)) # Subchunk2Size == NumSamples * NumChannels * BitsPerSample/8
  for x in audio.data:
    s.write(convertSample(x))

  s.close()
  let actualFilesize = getFileSize(filename)
  doAssert actualFilesize == expectedFilesize


if isMainModule:
  doAssert convertSample(+1.0) == +32767
  doAssert convertSample(-1.0) == -32768

  doAssert midiKeyToFreq(60).int == 261
  doAssert midiKeyToFreq(69).int == 440

  when false:
    let data = generateSine(440, 1.0, 1.0)
  else:
    let (data, _) = generateRandomNotes(5.0, 100)
  data.writeWave("test.wav")

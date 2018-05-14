import streams
import lenientops

import math
import os
import random
import sequtils
import strformat

import arraymancer

import audiotypes

randomize(seed=42)

type
  Data* = tuple[audio: AudioChunk, target: Tensor[float32]]
  NoteRange* = tuple[minMidiKey: int, maxMidiKey: int]

proc numNotes*(noteRange: NoteRange): int =
  noteRange.maxMidiKey - noteRange.minMidiKey + 1

proc keyAt*(noteRange: NoteRange, i: int): int =
  doAssert i < noteRange.numNotes
  noteRange.minMidiKey + i

const DEFAULT_NOTE_RANGE*: NoteRange = (minMidiKey: 21, maxMidiKey: 108)


proc midiKeyToFreq*(midiKey: int): float =
  ## Converts a midi key to frequency
  ## Reference: https://newt.phys.unsw.edu.au/jw/notes.html
  const concertPitchKey = 69
  const concertPitchFreq = 440.0
  let keyDiff = midiKey - concertPitchKey
  return pow(2, keyDiff / 12.0) * concertPitchFreq


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


proc generateSilenceLength*(sampleLength: int): AudioChunk =
  return AudioChunk(data: newSeq[SampleType](sampleLength))


proc generateSilence*(duration: float): AudioChunk =
  let sampleLength = int(duration * SAMPLE_RATE)
  return generateSilenceLength(sampleLength)


proc generateSine*(freq: float, amp: float, duration: float): AudioChunk =
  ## Note that this uses `sin` which means that generating a freq
  ## of SAMPLE_RATE/2 generates all zeros. If generating the frequency
  ## of SAMPLE_RATE/2 is required we could add an implemention using
  ## `cos` (which has the general drawback of being non-continuous at
  ## the first sample). Note that this would require to make adjustments
  ## to `amp` though, because the RMS would be too large when generating
  ## full -1 / +1 values.
  let sampleLength = int(duration * SAMPLE_RATE)
  let period = 1.0 / freq * SAMPLE_RATE

  var data = newSeq[SampleType](sampleLength)
  for i in 0 ..< sampleLength:
    data[i] = amp * sin(2 * PI * i / period)

  return AudioChunk(data: data)


proc generateRandomNotes*(duration: float, numNotes: int, minNoteLength = 0.1, maxNoteLength = 1.0, noteRange=DEFAULT_NOTE_RANGE): Data =
  var audio = generateSilence(duration)
  let maxIndex = audio.len - 1

  let (minMidiKey, maxMidiKey) = noteRange
  var groundTruth = newSeqWith(noteRange.numNotes, newSeq[float32](audio.len))

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


proc generateLinearNotes*(duration: float, noteRange=DEFAULT_NOTE_RANGE): Data =
  echo &"Time per note: {duration / noteRange.numNotes * 1000:.1f} ms"
  echo &"Period of lowest note: {1 / noteRange.minMidiKey.midiKeyToFreq * 1000:.1f} ms"

  var audio = generateSilence(duration)
  var groundTruth = newSeqWith(noteRange.numNotes, newSeq[float32](audio.len))

  for i in 0 ..< noteRange.numNotes:
    let sampleFrom = (audio.len / noteRange.numNotes * (i)).int
    let sampleUpto = (audio.len / noteRange.numNotes * (i+1)).int - 1
    let midiKey = noteRange.keyAt(i)
    audio.addNote(groundTruth[i], midiKey, sampleFrom, sampleUpto, envelopeLength=20)
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


when isMainModule:
  import strformat
  import algorithm

  doAssert convertSample(+1.0) == +32767
  doAssert convertSample(-1.0) == -32768

  doAssert midiKeyToFreq(60).int == 261
  doAssert midiKeyToFreq(69).int == 440

  block:
    # https://github.com/nim-lang/Nim/issues/7504
    const
      eps = 1.0e-7 ## Epsilon used for float comparisons.

    proc `~=`(x, y: float) =
      doAssert abs(x - y) < eps

    # SAMPLE_RATE/2 has a zero RMS because sine is all zeros
    generateSine(SAMPLE_RATE/2, 1.0, 1.0).rms ~= 0
    # but: SAMPLE_RATE/2 - 1 works
    generateSine(SAMPLE_RATE/2 - 1, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(SAMPLE_RATE/4, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(SAMPLE_RATE/8, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(SAMPLE_RATE/100, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(10, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(100, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(1000, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(10000, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(15000, 1.0, 1.0).rms ~= sqrt(0.5)
    generateSine(20000, 1.0, 1.0).rms ~= sqrt(0.5)

    let freqs = [
      SAMPLE_RATE/2,
      SAMPLE_RATE/2 - 1,
      20000,
      15000,
      SAMPLE_RATE/4 - 1,
      10000,
      SAMPLE_RATE/8 - 1,
      1000,
      100,
      10
    ]
    for f in freqs.reversed:
      let audio = generateSine(f, 1.0, 1.0)
      echo &"f: {f:10.2f}   rms: {audio.rms:10.6f}    meanAbs: {audio.meanAbs:10.6f}"

  when false:
    let data = generateSine(440, 1.0, 1.0)
  else:
    let (data, _) = generateRandomNotes(5.0, 100)
  data.writeWave("test.wav")

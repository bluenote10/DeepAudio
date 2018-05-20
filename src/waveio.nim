import streams
import strutils
import macros
import os

import audiotypes

template convertSample*(x: float): int16 =
  ## Converts a float in the range [-1.0, +1.0] to [-32768, +32767]
  doAssert(x <= +1.0)
  doAssert(x >= -1.0)
  int16(x * (high(int16).float + 0.5) - 0.5)

template convertSampleIntToFloat*(x: int16): float32 =
  ## Converts a float in the range [-32768, +32767] to [-1.0, +1.0]
  (x.float32 + 0.5) / 32767.5.float32


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


macro debug*(n: varargs[typed]): untyped =
  # `n` is a Nim AST that contains the whole macro invocation
  # this macro returns a list of statements:
  result = newNimNode(nnkStmtList, n)
  # iterate over any argument that is passed to this macro:
  for i in 0..n.len-1:
    # add a call to the statement list that writes the expression;
    # `toStrLit` converts an AST to its string representation:
    add(result, newCall("write", newIdentNode("stdout"), toStrLit(n[i])))
    # add a call to the statement list that writes ": "
    add(result, newCall("write", newIdentNode("stdout"), newStrLitNode(": ")))
    # add a call to the statement list that writes the expressions value:
    #add(result, newCall("writeln", newIdentNode("stdout"), n[i]))
    add(result, newCall("write", newIdentNode("stdout"), n[i]))
    # separate by ", "
    if i != n.len-1:
      add(result, newCall("write", newIdentNode("stdout"), newStrLitNode(", ")))

  # add new line
  add(result, newCall("writeLine", newIdentNode("stdout"), newStrLitNode("")))


proc swapEndian[T](x: T): T =
  result = x
  var arr = cast[array[sizeOf(T), char]](result)
  for i in 0 .. <sizeOf(T) div 2:
    swap(arr[i], arr[sizeOf(T) - 1 - i])

proc debugReadNBytes(s: Stream, n: int) =
  for i in 0 .. <n:
    debug s.readChar().BiggestInt.toHex(2)

proc skip(s: Stream, n: int) =
  var buffer = newSeq[char](n+1)
  discard s.readData(buffer[0].addr, n)

proc readLittleEndianUint16(s: Stream): uint16 =
  result = s.readInt16().uint16
  if system.cpuEndian != Endianness.littleEndian:
    result = swapEndian(result)

proc readLittleEndianUint32(s: Stream): uint32 =
  result = s.readInt32().uint32
  if system.cpuEndian != Endianness.littleEndian:
    result = swapEndian(result)

proc readLittleEndianInt24(s: Stream): int32 =
  let
    a = s.readInt8().uint8.uint32
    b = s.readInt8().uint8.uint32
    c = s.readInt8().uint8.uint32
  let temp = (
    a or
    b shl 8 or
    c shl 16
  )
  if (temp and 0x800000) == 0:
    result = cast[int32](temp)
  else:
    result = cast[int32](temp or 0xFF000000u32)


proc loadWave*(fn: string): AudioChunk =
  var s = newFileStream(fn, fmRead)

  if isNil(s):
    raise newException(IOError, "Cannot load stream from: " & fn)

  # RIFF header
  let riffStr = s.readStr(4)
  let remainingSizeAfter = s.readLittleEndianUint32
  let waveStr = s.readStr(4)
  debug riffStr
  debug remainingSizeAfter
  debug waveStr

  # fmt chunk
  let fmtHeader = s.readStr(4)
  let fmtRemainingSizeAfter = s.readLittleEndianUint32  #  will always be / must be 16
  let fmtFormatTag = s.readLittleEndianUint16
  let fmtNumChannels = s.readLittleEndianUint16
  let fmtSampleRate = s.readLittleEndianUint32
  let fmtBytesPerSec = s.readLittleEndianUint32
  let fmtBlockAlign = s.readLittleEndianUint16
  let fmtBitsPerSample = s.readLittleEndianUint16
  debug fmtHeader
  debug fmtRemainingSizeAfter
  debug fmtFormatTag
  debug fmtNumChannels
  debug fmtSampleRate
  debug fmtBytesPerSec
  debug fmtBlockAlign
  debug fmtBitsPerSample

  # data chunk
  while s.peekStr(4) != "data" and not s.atEnd():
    let tempHeader = s.readStr(4)
    let tempReamingSizeAfter = s.readLittleEndianUint32
    debug tempHeader
    debug tempReamingSizeAfter
    s.skip(tempReamingSizeAfter.int)

  let dataHeader = s.readStr(4)
  let dataReamingSizeAfter = s.readLittleEndianUint32
  debug dataHeader
  debug dataReamingSizeAfter

  var data = newSeq[float32]()

  var i = 0u32
  while i < dataReamingSizeAfter:
    let readVal = s.readInt16()
    data.add(convertSampleIntToFloat(readVal))
    i += 2

  s.close()
  result = AudioChunk(data: data)



proc loadWavePCM24*(fn: string): (seq[float32], seq[float32]) =
  var s = newFileStream(fn, fmRead)

  if isNil(s):
    raise newException(IOError, "Cannot load stream from: " & fn)

  # RIFF header
  let riffStr = s.readStr(4)
  let remainingSizeAfter = s.readLittleEndianUint32
  let waveStr = s.readStr(4)
  debug riffStr
  debug remainingSizeAfter
  debug waveStr

  # fmt chunk
  let fmtHeader = s.readStr(4)
  let fmtRemainingSizeAfter = s.readLittleEndianUint32  #  will always be / must be 16
  let fmtFormatTag = s.readLittleEndianUint16
  let fmtChannels = s.readLittleEndianUint16
  let fmtSampleRate = s.readLittleEndianUint32
  let fmtBytesPerSec = s.readLittleEndianUint32
  let fmtBlockAlign = s.readLittleEndianUint16
  let fmtBitsPerSample = s.readLittleEndianUint16
  debug fmtHeader
  debug fmtRemainingSizeAfter
  debug fmtFormatTag
  debug fmtChannels
  debug fmtSampleRate
  debug fmtBytesPerSec
  debug fmtBlockAlign
  debug fmtBitsPerSample

  # data chunk
  while s.peekStr(4) != "data" and not s.atEnd():
    let tempHeader = s.readStr(4)
    let tempReamingSizeAfter = s.readLittleEndianUint32
    debug tempHeader
    debug tempReamingSizeAfter
    s.skip(tempReamingSizeAfter.int)

  let dataHeader = s.readStr(4)
  let dataReamingSizeAfter = s.readLittleEndianUint32
  debug dataHeader
  debug dataReamingSizeAfter

  var channelL = newSeq[float32]()
  var channelR = newSeq[float32]()

  var i = 0u32
  while i < dataReamingSizeAfter:
    let readValL = s.readLittleEndianInt24()
    let readValR = s.readLittleEndianInt24()
    channelL.add readValL.float32 / 0x7FFFFF.float32
    channelR.add readValR.float32 / 0x7FFFFF.float32
    i += 6

  s.close()
  result = (channelL, channelR)


when isMainModule:
  import strformat
  import algorithm

  doAssert convertSample(+1.0) == +32767
  doAssert convertSample(-1.0) == -32768

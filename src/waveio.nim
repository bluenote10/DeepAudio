import streams
import strutils
import macros

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


proc loadWave*(fn: string): (seq[float32], seq[float32]) =
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



if isMainModule:
  echo system.cpuEndian
  discard loadWave("../../crass_action_closed_hat.wav")


import algorithm
import os
import cairo
import arraymancer
import lenientops
import sequtils
import sugar
import strformat
import strutils

import audiotypes
import waveio
import filters
import generator
import train_cnn

proc getQuantiles(data: TensorT, lower, upper: float): (float, float) =
  var flattenValues = data.data().sorted(system.cmp)
  let rankLower = (flattenValues.len() * lower).int
  let rankUpper = (flattenValues.len() * upper).int
  return (flattenValues[rankLower].float, flattenValues[rankUpper].float)


proc drawRoll*(data: TensorT, resX: int, resY: int, blockSizeY: int, fn: string) =
  var s = image_surface_create(FORMAT_ARGB32, resX.int32, resY.int32)
  var cr = s.create()

  let numCols = data.shape[0]
  let numRows = data.shape[1]

  let width = 0.8 * resX
  let height = 0.8 * resY

  let offsetX = ((resX - width) / 2).int
  let offsetY = ((resY - height) / 2).int

  cr.set_source_rgb(0x28 / 255, 0x2C / 255, 0x34 / 255)
  cr.paint()

  for col in 0 ..< numCols:
    for row in 0 ..< numRows:
      let value = data[col, row]
      let x1 = int(width / numCols * (col)) # + 0.5
      let x2 = int(width / numCols * (col+1)) # + 0.5
      let y1 = int(height / numRows * (row)) # + 0.5
      let y2 = int(height / numRows * (row+1)) # + 0.5
      let w = x2 - x1
      let h = y2 - y1
      cr.rectangle(
        0.5 + x1 + offsetX,
        0.5 + resY - y2 - offsetY,  # subtract Y since cooradinate system origin is top left
        0.0 + w,
        0.0 + h)
      cr.set_source_rgb(value, value, value)
      cr.fill()
      cr.set_line_width(1)
      #cr.set_source_rgb(1, 1, 1)
      cr.set_source_rgb(0x28, 0x2C, 0x34)
      cr.stroke()

  discard write_to_png(s, fn)
  cr.destroy()
  s.destroy()



proc visualizeRoll*(dataOrig: TensorT, outputDir: string) =
  # we extend data => clone required
  var data = dataOrig.clone()

  doAssert data.rank == 2
  let numKeys = data.shape[0]
  let N = data.shape[1]

  let (clipLower, clipUpper) = data.getQuantiles(0.01, 0.99)
  echo &"Writing images from output tensor of shape {data.shape} with min = {clipLower}, max = {clipUpper}"

  # geometry
  let resX = 1280
  let resY = 720
  let blockSizeY = 10
  let visibleBlocks = resY div blockSizeY

  let zerosBlock = zeros[TensorT.T]([data.shape[0], visibleBlocks-1])
  data = data.concat(zerosBlock, axis=1)
  echo data.shape

  # delete existing images to get clean imgage series
  discard execShellCmd(&"mkdir -p {outputDir}")
  discard execShellCmd(&"rm -f {outputDir}/img_*.png")

  for i in 0 ..< N:
    let idxFrom = i
    let idxUpto = i + visibleBlocks
    doAssert idxUpto <= data.shape[1]
    let subTensor = data[_, idxFrom..<idxUpto]

    stdout.write &"{i}\r"
    stdout.flushFile()
    drawRoll(subTensor, resX, resY, blockSizeY, &"{outputDir}/img_{i+1:010d}.png")


proc drawFretboard*(data: seq[SomeFloat], fn: string, numFrets=24, tuning=[5, 5, 5, 4, 5]) =
  let resX = 1200
  let resY = 200

  let numStrings = tuning.len + 1

  var s = image_surface_create(FORMAT_ARGB32, resX.int32, resY.int32)
  var cr = create(s)

  var tuningOffset = 0
  for i in 0 ..< numStrings:
    if tuningOffset >= data.len:
      break
    let stringData = data[tuningOffset ..< data.len]
    for j in 0 ..< numFrets:
      if j >= stringData.len:
        break
      let color = stringData[j]
      let x1 = resX / numFrets * (j)
      let x2 = resX / numFrets * (j+1)
      let y2 = resY - resY / numStrings * (i)
      let y1 = resY - resY / numStrings * (i+1)
      # echo x1, " ", x2, " ", y1, " ", y2, " ", i, " ", j
      cr.rectangle(x1, y1, x2-x1, y2-y1)
      cr.set_source_rgb(color, color, color)
      cr.fill()

    if i < tuning.len:
      tuningOffset += tuning[i]

  discard write_to_png(s, fn)
  destroy(cr)
  destroy(s)


proc draw*(tensor: TensorT, fn: string) =
  let resX = tensor.shape[1]
  let resY = tensor.shape[0]

  let min = tensor.min()
  let max = tensor.max()
  echo &"Drawing tensor with min = {min}, max = {max}"

  var s = image_surface_create(FORMAT_ARGB32, resX.int32, resY.int32)
  var cr = s.create()

  for i in 0 ..< tensor.shape[0]:
    for j in 0 ..< tensor.shape[1]:
      cr.rectangle(j.float64, i.float64, 1.float64, 1.float64)
      let color = tensor[i, j] # (tensor[i, j] - min) / (max - min)
      cr.set_source_rgb(color, color, color)
      cr.fill()

  discard write_to_png(s, fn)
  cr.destroy()
  s.destroy()


proc visualizeTensorSeq*(tensor: TensorT) =
  # TODO:
  # - make output dir configurable
  # - add video rendering

  let min = tensor.min()
  let max = tensor.max()
  echo &"Writing images from output tensor of shape {tensor.shape} with min = {min}, max = {max}"

  # delete existing images to get clean imgage series
  discard execShellCmd("rm -f imgs/img_*.png")

  var data = newSeq[float32](tensor.shape[0])
  for i in 0 ..< tensor.shape[1]:
    stdout.write &"{i}\r"
    stdout.flushFile()
    # let data = tensor[_, i].toRawSeq()
    for j in 0 ..< tensor.shape[0]:
      data[j] = tensor[j, i] # (tensor[j, i] - min) / (max - min)
    #drawRollSlice(data, &"imgs/img_{i:010d}.png")
    drawFretboard(data, &"imgs/img_{i+1:010d}.png")

#[
proc visualizeEnsemble*(audio: AudioChunk, chunkSize=512) =
  let minMidiKey = 21
  let maxMidiKey = 108

  var dataO = generateSilenceLength(audio.len)
  var output = newSeq[seq[float32]](maxMidiKey-minMidiKey+1)

  var ensemble = newSeq[TwoPole]()
  var i = 0
  for key in minMidiKey .. maxMidiKey:
    let f = key.midiKeyToFreq()
    var filter = twoPoleSearchPeak(f, 0.999) # twoPoleSearchPeak(f, 0.999)
    ensemble.add(filter)

    filter.process(audio, dataO)
    let rmsChunks = dataO.rmsChunks(chunkSize=chunkSize)
    output[i] = rmsChunks.normalized.data
    i += 1

    echo &"key = {key:5d}    f = {f:10.1f}    max = {rmsChunks.maxAbs:10.3f}    mean = {rmsChunks.mean:10.3f}"

  for i in 0 ..< output[0].len:
    let data = output.map(row => row[i].float)
    drawRollSlice(data, &"imgs/img_{i:010d}.png")
]#

when isMainModule:
  import random

  if true:
    let data = randomTensor(88, 10, 1'f32)
    visualizeRoll(data, ".test_img")

  if false:
    drawFretboard(@[0.1, 0.2, 0.3, 0.9, 1.0, 0.5], "test_fretboard_1.png")
    drawFretboard(newSeqWith(100, rand(1.0)), "test_fretboard_2.png")


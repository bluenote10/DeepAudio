import cairo
import arraymancer
import lenientops
import sequtils
import sugar
import strformat

import audiotypes
import waveio
import filters
import generator
import train_cnn


proc drawRollSlice*(data: seq[SomeFloat], fn: string) =
  let resX = 200.int32
  let resY = 600.int32

  var s = image_surface_create(FORMAT_ARGB32, resX, resY)
  var cr = create(s)

  for i, x in data:
    let y1 = resY / data.len * (i)
    let y2 = resY / data.len * (i+1)
    cr.rectangle(0.0, y1, resX.float, y2)
    let color = x
    cr.set_source_rgb(color, color, color)
    cr.fill()

  discard write_to_png(s, fn)
  destroy(cr)
  destroy(s)


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
  var cr = create(s)

  for i in 0 ..< tensor.shape[0]:
    for j in 0 ..< tensor.shape[1]:
      cr.rectangle(j.float64, i.float64, 1.float64, 1.float64)
      let color = tensor[i, j] # (tensor[i, j] - min) / (max - min)
      cr.set_source_rgb(color, color, color)
      cr.fill()

  discard write_to_png(s, fn)
  destroy(cr)
  destroy(s)


proc visualizeTensorSeq*(tensor: TensorT) =
  let min = tensor.min()
  let max = tensor.max()
  echo &"Writing images from output tensor of shape {tensor.shape} with min = {min}, max = {max}"

  var data = newSeq[float32](tensor.shape[0])
  for i in 0 ..< tensor.shape[1]:
    stdout.write &"{i}\r"
    stdout.flushFile()
    # let data = tensor[_, i].toRawSeq()
    for j in 0 ..< tensor.shape[0]:
      data[j] = tensor[j, i] # (tensor[j, i] - min) / (max - min)
    #drawRollSlice(data, &"imgs/img_{i:010d}.png")
    drawFretboard(data, &"imgs/img_{i:010d}.png")


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


when isMainModule:
  import random

  if false:
    drawRollSlice(@[0.1, 0.2, 0.3, 0.9, 1.0, 0.5], "test.png")

  if true:
    drawFretboard(@[0.1, 0.2, 0.3, 0.9, 1.0, 0.5], "test_fretboard_1.png")
    drawFretboard(newSeqWith(100, rand(1.0)), "test_fretboard_2.png")


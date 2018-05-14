import sequtils
import sugar
import math
import strformat

import audiotypes
import filters
import generator

import arraymancer
import matplotlib


proc toSeq2D*[T](t: Tensor[T]): seq[seq[T]] =
  if t.rank != 2:
    raise newException(ValueError, "Tensor must be of rank 2")

  result = newSeqWith(
    t.shape[0], newSeq[T](t.shape[1])
  )
  for i in 0 ..< t.shape[0]:
    for j in 0 ..< t.shape[1]:
      result[i][j] = t[i, j]
  #[
  # Doesn't work because of https://github.com/nim-lang/Nim/issues/7816
  result = toSeq(0 .. t.shape[0]).map(i =>
    t[i, _].toRawSeq
  )
  ]#


proc toSeq3D*[T](t: Tensor[T]): seq[seq[seq[T]]] =
  if t.rank != 3:
    raise newException(ValueError, "Tensor must be of rank 3")

  #[
  result = newSeqWith(
    t.shape[0], newSeqWith(
      t.shape[1], newSeq[T](t.shape[2])
    )
  )
  ]#
  #[
  result = toSeq(0 .. t.shape[0]).map(i =>
    toSeq(0 .. t.shape[1]).map(j =>
      t[i, j, _].toRawSeq
    )
  )
  ]#
  # TODO ...


proc plotTensor[T](t: seq[seq[T]]) =
  var p = createSinglePlot()
  p.imshow(t, aspect := "auto", interpolation := "nearest")
  p.colorbar()
  p.show()
  p.run()


proc processEnsemble(data: Data) =
  let minMidiKey = 21
  let maxMidiKey = 108

  var dataO = generateSilenceLength(data.audio.len)
  #var output = zeros[float32]([maxMidiKey-minMidiKey+1, data.audio.len])
  var output = newSeq[seq[float32]](maxMidiKey-minMidiKey+1)

  var ensemble = newSeq[TwoPole]()
  var i = 0
  for key in minMidiKey .. maxMidiKey:
    let f = key.midiKeyToFreq()
    var filter = twoPoleSearchPeak(f, 0.999) # twoPoleSearchPeak(f, 0.999)
    ensemble.add(filter)

    filter.process(data.audio, dataO)
    #dataO.normalize()
    #output[i, _] = dataO.data
    let rmsChunks = dataO.rmsChunks(chunkSize=512)
    output[i] = rmsChunks.normalized.data
    i += 1

    echo &"key = {key:5d}    f = {f:10.1f}    max = {rmsChunks.maxAbs:10.3f}    mean = {rmsChunks.mean:10.3f}"

  #plotTensor(data.target.toSeq2D)
  #plotTensor(output.toSeq2D)
  plotTensor(output)



when isMainModule:
  #let data = generateRandomNotes(10.0, 100)
  let data = generateLinearNotes(20.0)
  data.audio.writeWave("linear.wav")
  processEnsemble(data)



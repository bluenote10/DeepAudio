import arraymancer
import strformat
import math
import lenientops

import audiotypes
import generator
import filters

type
  TensorT = Tensor[SampleType]
  Dataset = tuple[x_train: Variable[TensorT], y_train: TensorT]


proc processEnsemble(audio: AudioChunk, chunkSize=512, noteRange=DEFAULT_NOTE_RANGE): TensorT =
  let outputLength = (audio.len / chunkSize).ceil.int
  result = zeros[SampleType](noteRange.numNotes, outputLength)

  var dataO = generateSilenceLength(audio.len)

  var ensemble = newSeq[TwoPole]()
  for key in noteRange.minMidiKey .. noteRange.maxMidiKey:
    let i = noteRange.indexOf(key)
    let f = key.midiKeyToFreq()
    var filter = twoPoleSearchPeak(f, 0.999) # twoPoleSearchPeak(f, 0.999)
    ensemble.add(filter)

    filter.process(audio, dataO)
    let rmsChunks = dataO.rmsChunks(chunkSize=chunkSize)
    result[i, _] = rmsChunks.normalized.data


proc processGroundTruth(truth: TensorT, chunkSize=512): TensorT =
  let numKeys = truth.shape[0]
  let numChunks = (truth.shape[1] / chunkSize).ceil.int
  result = zeros[SampleType](numKeys, numChunks)

  for keyIndex in 0 ..< numKeys:
    #[
    var i = 0
    var chunk = 0
    var ampSum = 0.0
    while i < truth.shape[1]:
      if i % chunkSize == chunkSize - 1:

      ampSum += truth[]
    ]#
    proc limited(x, limit: int): int =
      if x > limit: limit else: x

    for chunkIndex in 0 ..< numChunks:
      let chunkFrom = chunksize * chunkIndex
      let chunkUpto = (chunksize * (chunkIndex + 1)).limited(truth.shape[1])
      var ampSum = 0.0
      for i in chunkFrom ..< chunkUpto:
        ampSum += truth[keyIndex, i]
      let realChunkSize = chunkUpto - chunkFrom
      let rms = (ampSum / realChunkSize) / sqrt(2.0)
      #echo &"{chunkFrom} {chunkUpto} {realChunkSize} {rms}"
      result[keyIndex, chunkIndex] = rms

proc lossMSE(a, b: TensorT): float =
  doAssert a.rank == 2
  doAssert b.rank == 2
  doAssert a.shape == b.shape
  var sum = 0.0
  for i in 0 ..< a.shape[0]:
    for j in 0 ..< a.shape[1]:
      let diff = a[i, j] - b[i, j]
      sum += diff*diff
  let N = a.shape[0] * a.shape[1]
  return sum / N


proc loadData(): Dataset =
  let data = generateRandomNotes(5.0, 100)

  let chunkSize = 128
  let X = processEnsemble(data.audio, chunkSize)
  let Y = processGroundTruth(data.truth, chunkSize)
  echo X.shape
  echo Y.shape
  echo X.mean
  echo Y.mean
  echo &"loss = {lossMSE(Y, Y.zeros_like()):8.6f} (zero prediction)"
  echo &"loss = {lossMSE(Y, Y.ones_like() * Y.mean):8.6f} (mean prediction)"
  echo &"loss = {lossMSE(Y, X):8.6f} (resonators unscaled)"
  echo &"loss = {lossMSE(Y, X * Y.mean / X.mean):8.6f} (resonators mean scaled)"


proc train() =
  let ctx = newContext(TensorT)


when isMainModule:
  discard loadData()

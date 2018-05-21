import arraymancer
import strformat
import math
import lenientops
import sequtils

import audiotypes
import generator
import filters

import matplotlib

type
  TensorT = Tensor[SampleType]
  VariableT = Variable[TensorT]
  #Dataset = tuple[x_train: Variable[TensorT], y_train: TensorT]
  Dataset = tuple[X: TensorT, Y: TensorT]


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
    result[i, _] = rmsChunks.normalized.data.toTensor.unsqueeze(0)


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
  return (X, Y)


iterator batchGenerator(X: TensorT, Y: TensorT, batchSize=4, seqSize=3): (TensorT, TensorT) =
  let numKeys = X.shape[0]
  var batchX = zeros[float32](batchSize, seqSize, numKeys)
  var batchY = zeros[float32](batchSize, numKeys)

  var i = 0 # == start index of batch
  block whileLoop:
    while true:
      for j in 0 ..< batchSize:
        if i+j+seqSize > X.shape[1]: # `>` is correct because i+j+seqSize is exclusive
          break whileLoop
        batchX[j, _, _] = X[_, (i+j)..<(i+j+seqSize)].transpose().unsqueeze(0)
        batchY[j, _] = Y[_, i+j+seqSize-1].transpose()
      yield (batchX, batchY)
      i += batchSize


proc model_fc_forward(X, lin1, lin2: VariableT): VariableT =
  let hidden = X.flatten().linear(lin1).relu()
  result = hidden.linear(lin2)
  if false:
    echo &"shape of x:      {X.value.shape}"
    echo &"shape of x:      {X.flatten().value.shape}"
    echo &"shape of hidden: {hidden.value.shape}"
    echo &"shape of result: {result.value.shape}"


proc train_fc(data: Dataset, numHidden=200, seqLength=2) =
  let ctx = newContext(TensorT)

  let numKeys = data.X.shape[0]
  let numInput = numKeys * seqLength

  let
    lin1 = ctx.variable(
      randomTensor(numHidden, numInput, 1'f32) .- 0.5'f32,
      requires_grad = true
    )
    lin2 = ctx.variable(
      randomTensor(numKeys, numHidden, 1'f32) .- 0.5'f32,
      requires_grad = true
    )

  let optim = newSGD[float32](
    lin1, lin2, 0.0001f
  )

  var losses = newSeq[float]()

  # Learning loop
  for epoch in 1 .. 100:
    var lossInEpoch = 0.0
    for batchX, batchY in batchGenerator(data.X, data.Y, batchSize=32, seqSize=seqLength):

      # Running through the network and computing loss
      var batchXVar = ctx.variable(batchX)
      let output = batchXVar.model_fc_forward(lin1, lin2)
      let loss = output.mse_loss(batchY)
      let lossScalar = loss.value[0]
      # echo &"{batchX.shape} {output.value.shape} {batchY.shape}"
      # echo &"epoch = {epoch:3d}    loss = {lossScalar:10.6f}"
      losses.add(lossScalar)
      lossInEpoch += lossScalar

      # Compute the gradient (i.e. contribution of each parameter to the loss)
      loss.backprop()

      # Correct the weights now that we have the gradient information
      optim.update()
    echo &"loss = {lossInEpoch:10.6f}"

  var p = createSinglePlot()
  p.plot(toSeq(1 .. losses.len), losses, "-o")
  p.show()
  p.run()

when isMainModule:

  if false: # TODO: write proper test...
    var X: TensorT = zeros[float32](5, 14)
    for i in 0 ..< X.shape[0]:
      for j in 0 ..< X.shape[1]:
        X[i, j] = j.float32 # + (i / 10).float32
    echo &"X = {X}"

    var Y: TensorT = zeros[float32](5, 14)
    for i in 0 ..< X.shape[0]:
      for j in 0 ..< X.shape[1]:
        Y[i, j] = -j.float32 # + (i / 10).float32
    echo &"Y = {Y}"

    var i = 0
    for batchX, batchY in batchGenerator(X, Y):
      echo &"batchX = {batchX}"
      echo &"batchY = {batchY}"
      i += 1

    echo &"Num batches: {i}"

  if true:
    let data = loadData()
    train_fc(data)


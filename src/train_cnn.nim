import arraymancer
import strformat
import math
import lenientops
import sequtils
import sugar
import os

import audiotypes
import datatypes
import generator
import filters
import waveio

import matplotlib
import visualization


# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------

proc processEnsemble*(audio: AudioChunk, chunkSize=512, noteRange=DEFAULT_NOTE_RANGE): TensorT =
  let outputLength = (audio.len / chunkSize).ceil.int
  result = zeros[SampleType](noteRange.numNotes, outputLength)

  var dataO = generateSilenceLength(audio.len)

  var ensemble = newSeq[TwoPole]()
  for key in noteRange.minMidiKey .. noteRange.maxMidiKey:
    let i = noteRange.indexOf(key)
    let f = key.midiKeyToFreq()
    var filter = twoPoleSearchPeak(f, 0.999) # twoPoleSearchPeak(f, 0.999)
    echo &"f = {f:6.1f}    {filter}"
    ensemble.add(filter)

    filter.process(audio, dataO)
    let rmsChunks = dataO.rmsChunks(chunkSize=chunkSize)
    result[i, _] = rmsChunks.normalized.data.toTensor.unsqueeze(0)


proc processGroundTruth*(truth: TensorT, chunkSize=512): TensorT =
  let numKeys = truth.shape[0]
  let numChunks = (truth.shape[1] / chunkSize).ceil.int
  result = zeros[SampleType](numKeys, numChunks)

  for keyIndex in 0 ..< numKeys:
    proc limited(x, limit: int): int =
      if x > limit: limit else: x

    for chunkIndex in 0 ..< numChunks:
      let chunkFrom = chunksize * chunkIndex
      let chunkUpto = (chunksize * (chunkIndex + 1)).limited(truth.shape[1])
      var ampSum = 0.0
      for i in chunkFrom ..< chunkUpto:
        ampSum += truth[keyIndex, i]
      let realChunkSize = chunkUpto - chunkFrom
      # TODO: What convention should we use here?
      # Does the truth vector contain amplitudes or RMS values?
      # Currently they are amplitudes, but this has the drawback
      # that we must make an assumption of the underlying waveform
      # to convert to RMS (dividing by sqrt(2) only appropriate for sine).
      # Maybe it would be better to assumate the we have RMS values
      # already.
      let rms = (ampSum / realChunkSize) # / sqrt(2.0)
      #echo &"{chunkFrom} {chunkUpto} {realChunkSize} {rms}"
      result[keyIndex, chunkIndex] = rms

# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------

proc lossMSE*(a, b: TensorT): float =
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


proc showReferenceLosses*(X, Y, P: TensorT) =
  ## Compares the loss of a prediction to some trivial estimators
  let meanX = X.mean
  let meanY = Y.mean
  let meanP = P.mean
  echo &"[X] min: {X.min():8.3f}    mean: {meanX:8.3f}    max: {X.max():8.3f}"
  echo &"[Y] min: {Y.min():8.3f}    mean: {meanY:8.3f}    max: {Y.max():8.3f}"
  echo &"[P] min: {P.min():8.3f}    mean: {meanP:8.3f}    max: {P.max():8.3f}"

  # simple linear regression
  let m = ((X .- meanX) .* (Y .- meanY)).sum() / ((X .- meanX) .* (X .- meanX)).sum()
  let c = meanY - m * meanX

  let lossZeroPrediction = lossMSE(Y, Y.zeros_like())
  let lossMeanPrediction = lossMSE(Y, Y.ones_like() * meanY)
  let lossResonatorsUnscaled = lossMSE(Y, X)
  let lossResonatorsScaled = lossMSE(Y, X * meanY / meanX)
  let lossResonatorsLinReg = lossMSE(Y, m*X .+ c)
  let lossPrediction = lossMSE(Y, P)
  echo &"loss = {lossZeroPrediction:8.6f} (zero prediction)"
  echo &"loss = {lossMeanPrediction:8.6f} (mean prediction)"
  echo &"loss = {lossResonatorsUnscaled:8.6f} (resonators unscaled)"
  echo &"loss = {lossResonatorsScaled:8.6f} (resonators mean scaled)"
  echo &"loss = {lossResonatorsLinReg:8.6f} (resonators linear regression)"
  echo &"loss = {lossPrediction:8.6f} (model)"

# -----------------------------------------------------------------------------
# Batch generator
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Simple fully connected model
# -----------------------------------------------------------------------------

type
  ModelFC* = object
    lin1*, lin2*, bias1*, bias2*: TensorT
    numKeys*, seqLength*: int

proc showVars*(m: ModelFC) =
  echo &"Shape lin1: {m.lin1.shape}"
  echo &"Shape lin2: {m.lin2.shape}"
  echo &"Shape bias1: {m.bias1.shape}"
  echo &"Shape bias2: {m.bias2.shape}"
  echo &"numKeys: {m.numKeys}"
  echo &"seqLength: {m.seqLength}"


proc model_fc_forward(X, lin1, lin2, bias1, bias2: VariableT): VariableT =
  let hidden = X.flatten().linear(lin1, bias1) # .relu()
  result = hidden.linear(lin2, bias2)
  if false:
    echo &"shape of x:      {X.value.shape}"
    echo &"shape of x:      {X.flatten().value.shape}"
    echo &"shape of hidden: {hidden.value.shape}"
    echo &"shape of result: {result.value.shape}"


proc predict_fc*(model: ModelFC, X: TensorT): TensorT =
  let ctx = newContext(TensorT)
  let batchSize = 32

  var output = X.zeros_like()

  # Note: it is crucial that we write the batch outputs with the proper
  # offset into the final output. Since the model is trained on the Y
  # slice corresponding to the last index of the sequence, we have to
  # start at the index (it's not possible to output anything meaningful
  # for time indices before this index). From there, we can jump in
  # multiples of batchSize.
  var i = model.seqLength - 1

  let dummyY = X.zeros_like() # TODO: Y should be optional for batchGenerator?
  for batchX, batchY in batchGenerator(X, dummyY, batchSize=batchSize, seqSize=model.seqLength):
    var batchXVar = ctx.variable(batchX)
    let lin1 = ctx.variable(model.lin1)
    let lin2 = ctx.variable(model.lin2)
    let bias1 = ctx.variable(model.bias1)
    let bias2 = ctx.variable(model.bias2)
    let batchOutput = batchXVar.model_fc_forward(lin1, lin2, bias1, bias2)
    output[_, i..<i+batchSize] = batchOutput.value.transpose()
    i += batchSize

  return output


proc train_fc*(dataGen: DatasetGen, numDatasets=5, numEpochs=1000, numKeys=88, numHidden=500, seqLength=2): ModelFC =
  let ctx = newContext(TensorT)

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
    bias1 = ctx.variable(
      randomTensor(1, numHidden, 1'f32) .- 0.5'f32,
      requires_grad = true
    )
    bias2 = ctx.variable(
      randomTensor(1, numKeys, 1'f32) .- 0.5'f32,
      requires_grad = true
    )

  proc wrapModel(): ModelFC = ModelFC(
    lin1: lin1.value,
    lin2: lin2.value,
    bias1: bias1.value,
    bias2: bias2.value,
    numKeys: numKeys,
    seqLength: seqLength
  )

  let optim = newSGD[float32](
    lin1, lin2, bias1, bias2, 0.000005f
  )

  var losses = newSeq[float]()

  # Learning loop
  for datasetId in 1 .. numDatasets:
    echo &"\n *** Training on dataset {datasetId}:"

    let data = dataGen(datasetId)
    doAssert numKeys == data.X.shape[0]

    for epoch in 1 .. numEpochs:
      var lossInEpoch = 0.0
      var batchCount = 0

      for batchX, batchY in batchGenerator(data.X, data.Y, batchSize=32, seqSize=seqLength):
        batchCount += 1

        # Running through the network and computing loss
        var batchXVar = ctx.variable(batchX)
        let output = batchXVar.model_fc_forward(lin1, lin2, bias1, bias2)
        let loss = output.mse_loss(batchY)
        let lossScalar = loss.value[0]
        # echo &"{batchX.shape} {output.value.shape} {batchY.shape}"
        # echo &"epoch = {epoch:3d}    loss = {lossScalar:10.6f}"
        losses.add(lossScalar)
        lossInEpoch += lossScalar

        # backprop + update
        loss.backprop()
        optim.update()

      if epoch == 1:
        echo &"loss: {lossInEpoch / batchCount:10.6f}    [initial]"
      stdout.write &"loss: {lossInEpoch / batchCount:10.6f}    epoch: {epoch:6d}    [num batches: {batchCount}]\r"
      stdout.flushFile()

    let pred = wrapModel().predict_fc(data.X)
    echo &"\n *** Results for dataset {datasetId}:"
    showReferenceLosses(data.X, data.Y, pred)
    data.X.draw(&"train_{datasetId:03d}_X.png")
    data.Y.draw(&"train_{datasetId:03d}_Y.png")
    pred.draw(&"train_{datasetId:03d}_P.png")

  echo "Plotting losses..."
  var p = createSinglePlot()
  p.plot(toSeq(1 .. losses.len), losses, "-")
  p.yscaleLog()
  p.saveFigure("losses.png")
  #p.show()
  p.run()

  result = wrapModel()


proc train_py*(dataGen: DatasetGen, numDatasets=5, numEpochs=1000, numKeys=88, numHidden=500, seqLength=2): ModelFC =

  # Learning loop
  for datasetId in 1 .. numDatasets:
    echo &"\n *** Training on dataset {datasetId}:"

    let data = dataGen(datasetId)
    doAssert numKeys == data.X.shape[0]

    data.X.write_npy("X.npy")
    data.Y.write_npy("Y.npy")

    doAssert execShellCmd("rm -f test.model") == 0
    doAssert execShellCmd("./pymodels/model.py --model test.model") == 0

    let pred = read_npy[SampleType]("P.npy")
    echo &"\n *** Results for dataset {datasetId}:"
    showReferenceLosses(data.X, data.Y, pred)
    data.X.draw(&"train_{datasetId:03d}_X.png")
    data.Y.draw(&"train_{datasetId:03d}_Y.png")
    pred.draw(&"train_{datasetId:03d}_P.png")



proc predict_py*(X: TensorT): TensorT =
  X.write_npy("X.npy")
  doAssert execShellCmd("./pymodels/model.py --model test.model --predict-only") == 0
  let pred = read_npy[SampleType]("P.npy")
  pred


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


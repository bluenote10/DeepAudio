import arraymancer

import lenientops
import sequtils
import sugar
import strformat
import os

import audiotypes
import datatypes
import waveio
import filters
import generator
import train_cnn
import visualization
import serialization


proc loadData*(datasetId: int, chunkSize: int): Dataset =
  let length = 60.0

  let data =
    if datasetId mod 2 == 1:
      generateRandomNotes(length, 500)
    else:
      generateLinearNotes(length)

  data.audio.writeWave("training_data.wav")

  let X = processEnsemble(data.audio, chunkSize)
  let Y = processGroundTruth(data.truth, chunkSize)
  doAssert X.shape == Y.shape
  return (X, Y)


proc main() =
  let args = commandLineParams()
  if args.len == 0:
    echo "Error: no mode specified"
    quit(1)

  let modes = args
  let mode = args[0]

  let chunkSize = 44100 div 30
  echo &"Using chunksize of {chunkSize} samples, corresponding to chunks of {1000.0 * chunkSize / 44100.0} ms"

  # TODO: Replace this by loadWave => createEnsemble => visualizeRoll
  #if mode == "resonator_test":
  #  let data = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
  #  visualizeEnsemble(data, chunkSize=44100 div 30)

  if mode == "train_test":
    let data = loadData(datasetId=1, chunkSize=chunkSize)
    let model = train_fc((datasetId) => data, numDatasets=1, numEpochs=100)
    let prediction = model.predict_fc(data.X)

    data.X.draw("data_X.png")
    data.Y.draw("data_Y.png")
    prediction.draw("data_pred.png")
    showReferenceLosses(data.X, data.Y, prediction)

  var model: ModelFC

  if "train" in modes:
    model = train_fc((datasetId) => loadData(datasetId, chunkSize=chunkSize), numDatasets=10, numEpochs=500)
    model.showVars()
    model.storeAsFile("models/model.dat")

    # create some data for out-of-sample validation
    for datasetId in 1 .. 2:
      echo &"\n *** Validating on dataset {datasetId}:"
      let data = loadData(datasetId, chunkSize=chunkSize)
      let pred = model.predict_fc(data.X)
      showReferenceLosses(data.X, data.Y, pred)

      data.X.draw(&"test_{datasetId:03d}_X.png")
      data.Y.draw(&"test_{datasetId:03d}_Y.png")
      pred.draw(&"test_{datasetId:03d}_P.png")

  if "test" in modes:
    model = restoreFromFile("models/model.dat", ModelFC)
    model.showVars()

    let audio = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    let dataF = processEnsemble(audio, chunkSize=chunkSize)
    let predictionF = model.predict_fc(dataF)

    dataF.draw("pitches_resonators.png")
    predictionF.draw("pitches_prediction.png")
    #visualizeTensorSeq(predictionF)
    visualizeRoll(predictionF, "./imgs")

  if "train_py" in modes:
    model = train_py((datasetId) => loadData(datasetId, chunkSize=chunkSize), numDatasets=11, numEpochs=500)

    # create some data for out-of-sample validation
    for datasetId in 1 .. 2:
      echo &"\n *** Validating on dataset {datasetId}:"
      let data = loadData(datasetId, chunkSize=chunkSize)
      let pred = predict_py(data.X)
      showReferenceLosses(data.X, data.Y, pred)

      data.X.draw(&"test_{datasetId:03d}_X.png")
      data.Y.draw(&"test_{datasetId:03d}_Y.png")
      pred.draw(&"test_{datasetId:03d}_P.png")

  if "test_py" in modes:
    let audio = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    let dataF = processEnsemble(audio, chunkSize=chunkSize)
    let predictionF = predict_py(dataF)

    dataF.draw("pitches_resonators.png")
    predictionF.draw("pitches_prediction.png")
    #visualizeTensorSeq(predictionF)
    visualizeRoll(predictionF, "./imgs")

when isMainModule:
  main()

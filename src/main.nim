import arraymancer

import lenientops
import sequtils
import sugar
import strformat
import os

import audiotypes
import waveio
import filters
import generator
import train_cnn
import visualization
import serialization


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
    let data = loadData(chunkSize=chunkSize)
    let model = train_fc(() => data, numDatasets=1, numEpochs=100)
    let prediction = model.predict_fc(data.X)

    data.X.draw("data_X.png")
    data.Y.draw("data_Y.png")
    prediction.draw("data_pred.png")
    showReferenceLosses(data.X, data.Y, prediction)

  var model: ModelFC

  if "train" in modes:
    model = train_fc(() => loadData(chunkSize=chunkSize), numDatasets=10, numEpochs=500)
    model.showVars()

    # create some test data
    let dataT = loadData(chunkSize=chunkSize)
    let predictionT = model.predict_fc(dataT.X)

    dataT.X.draw("data_X.png")
    dataT.Y.draw("data_Y.png")
    predictionT.draw("data_pred.png")
    showReferenceLosses(dataT.X, dataT.Y, predictionT)

    model.storeAsFile("models/model.dat")

  if "test" in modes:
    model = restoreFromFile("models/model.dat", ModelFC)
    model.showVars()

    let audio = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    let dataF = processEnsemble(audio, chunkSize=chunkSize)
    let predictionF = model.predict_fc(dataF)

    dataF.draw("pitches_resonators.png")
    predictionF.draw("pitches_prediction.png")
    visualizeTensorSeq(predictionF)

  if "train_py" in modes:
    model = train_py(() => loadData(chunkSize=chunkSize), numDatasets=10, numEpochs=500)

    # create some test data
    let dataT = loadData(chunkSize=chunkSize)
    let predictionT = predict_py(dataT.X)

    dataT.X.draw("data_X.png")
    dataT.Y.draw("data_Y.png")
    predictionT.draw("data_pred.png")
    showReferenceLosses(dataT.X, dataT.Y, predictionT)

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

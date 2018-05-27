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

  if mode == "train_test":
    let data = loadData(chunkSize=512)
    let model = train_fc(() => data)
    let prediction = model.predict_fc(data.X)
    showReferenceLosses(data.X, data.Y, prediction)

  if mode == "resonator_test":
    let data = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    visualizeEnsemble(data, chunkSize=44100 div 30)

  var model: ModelFC

  if "train" in modes:
    let chunkSize = 44100 div 30
    model = train_fc(() => loadData(chunkSize=chunkSize))
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
    let chunkSize = 44100 div 30
    model = restoreFromFile("models/model.dat", ModelFC)
    model.showVars()

    let audio = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    let dataF = processEnsemble(audio, chunkSize=chunkSize)
    let predictionF = model.predict_fc(dataF)

    dataF.draw("pitches_resonators.png")
    predictionF.draw("pitches_prediction.png")
    visualizeTensorSeq(predictionF)


when isMainModule:
  main()

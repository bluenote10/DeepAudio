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
import visualization

when isMainModule:

  if false:
    let data = loadData(chunkSize=512)
    let model = train_fc(() => data)
    let prediction = model.predict_fc(data.X)
    showReferenceLosses(data.X, data.Y, prediction)

  if false:
    let data = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    visualizeEnsemble(data, chunkSize=44100 div 30)

  if true:
    let chunkSize = 44100 div 30

    let model = train_fc(() => loadData(chunkSize=chunkSize))

    # create some test data
    let dataT = loadData(chunkSize=chunkSize)
    let predictionT = model.predict_fc(dataT.X)

    dataT.X.draw("data_X.png")
    dataT.Y.draw("data_Y.png")
    predictionT.draw("data_pred.png")
    showReferenceLosses(dataT.X, dataT.Y, predictionT)

    let audio = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    let dataF = processEnsemble(audio, chunkSize=chunkSize)
    let predictionF = model.predict_fc(dataF)

    dataF.draw("pitches_resonators.png")
    predictionF.draw("pitches_prediction.png")
    visualizeTensorSeq(predictionF)


    # ffmpeg -r 30 -i imgs/img_%010d.png -i audio/Sierra\ Hull\ \ Black\ River\ \(OFFICIAL\ VIDEO\).wav -c:v libx264 -c:a aac -pix_fmt yuv420p -crf 23 -r 30 -strict -2 -shortest -y video-from-frames.mp4

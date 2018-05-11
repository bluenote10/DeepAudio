import audiotypes
# TODO move into eval module:
import generator
import matplotlib

import math
import lenientops
import strformat
import sugar


type
  TwoPole = object
    b0, a1, a2: SampleType
    y_1, y_2: SampleType


proc process*(filter: var TwoPole, dataI: AudioChunk, dataO: var AudioChunk) =
  ## Model according two: https://ccrma.stanford.edu/~jos/fp/Two_Pole.html
  assert dataI.len == dataO.len

  let b0 = filter.b0
  let a1 = filter.a1
  let a2 = filter.a2

  var y_1 = filter.y_1
  var y_2 = filter.y_2

  var y: SampleType
  for i in 0 ..< dataI.len:
    let x = dataI.data[i]
    y = b0*x - a1*y_1 - a2*y_2
    y_2 = y_1
    y_1 = y
    # echo y
    dataO.data[i] = y

  filter.y_1 = y_1
  filter.y_2 = y_2


proc reset*(filter: var TwoPole) =
  filter.y_1 = 0
  filter.y_2 = 0


proc twoPole(freq: float, R: float): TwoPole =
  let T = 1.0 / SAMPLE_RATE # sampling interval
  let theta_c = 2 * PI * freq * T

  let a1: SampleType = -2 * R * cos(theta_c)
  let a2: SampleType = R^2

  let b0: SampleType = 0.1 # ?

  echo &"a1: ${a1}, a2: ${a2}, b0: ${b0}, theta_c: ${theta_c}, T: ${T}"
  TwoPole(b0: b0, a1: a1, a2: a2, y_1: 0, y_2: 0)


proc plotMeasurement(xs: seq[float], ys: seq[float]) =
  var p = createSinglePlot()
  p.plot(xs, ys, "-")
  p.show()
  p.run()


proc measureFilter*[Filter](filter: var Filter) =
  let f1 = 20f
  let f2 = 10000f
  let steps = 1000
  echo &"Resolution: {(f2-f1) / steps} hz"

  const duration = 1.0

  var dataO = generateSilence(duration)

  var measurementsX = newSeqOfCap[float](steps)
  var measurementsY = newSeqOfCap[float](steps)

  var peakF = 0.0
  var peakA = 0.0

  for i in 0 .. steps:
    let f = f1 + (f2-f1) / steps * i
    let dataI = generateSine(f, 1.0, duration)
    filter.reset()
    filter.process(dataI, dataO)
    let amp = dataO.meanAbs()
    measurementsX.add(f)
    measurementsY.add(amp)
    if amp > peakA:
      peakF = f
      peakA = amp

  echo &"Peak frequency: {peakF} hz (amplitude: {peakA})"
  plotMeasurement(measurementsX, measurementsY)

when isMainModule:

  var filter = twoPole(100, 0.99)
  measureFilter(filter)
import audiotypes
# TODO move into eval module:
import generator
import matplotlib

import math
import complex
import lenientops
import strformat
import sugar


proc goldenSectionSearch*(f: (float) -> float; a, b: float; tol=1e-5): float =
  ## https://en.wikipedia.org/wiki/Golden-section_search
  const goldenRatio = (sqrt(5.0) + 1) / 2
  var a = a
  var b = b
  var c = b - (b - a) / goldenRatio
  var d = a + (b - a) / goldenRatio

  while abs(c - d) > tol:
    # echo &"{c} {d}"
    if f(c) < f(d):
      b = d
    else:
      a = c
    # we recompute both c and d here to avoid loss of precision which may lead to incorrect results or infinite loop
    c = b - (b - a) / goldenRatio
    d = a + (b - a) / goldenRatio

  return (b + a) / 2


type
  TwoPole* = object
    b0, a1, a2: SampleType
    y_1, y_2: SampleType


proc process*(filter: var TwoPole, dataI: AudioChunk, dataO: var AudioChunk) =
  ## Model according two: https://ccrma.stanford.edu/~jos/fp/Two_Pole.html
  ##
  ## Wolfram Alpha expression:
  ## b / (1 + a_1 * e^(-i w T) + a_2 * e^(-2 * i * w * T))
  ## | a_1 cos(w T) + i a_1 sin(w T) + a_2 cos(2 w T) + i a_2 sin(2 w T) |
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


proc response*(filter: TwoPole, freq: float): Complex =
  let i: Complex = (re: 0.0, im: 1.0)
  let omega = 2 * PI * freq
  let T = 1.0 / SAMPLE_RATE
  let b0 = filter.b0
  let a1 = filter.a1
  let a2 = filter.a2
  result = b0 / (1 + a1*exp(-i*omega*T) + a2*exp(-i*2*omega*T))


proc peakFreq*(filter: TwoPole): float =
  let f = goldenSectionSearch((f: float) => -filter.response(f).abs(), 0, SAMPLE_RATE/2)
  f


proc twoPole*(freq: float, R: float): TwoPole =
  let T = 1.0 / SAMPLE_RATE # sampling interval
  let theta_c = 2 * PI * freq * T

  let a1: SampleType = -2 * R * cos(theta_c)
  let a2: SampleType = R^2

  let b0: SampleType = 1.0 # ?

  # echo &"a1: ${a1}, a2: ${a2}, b0: ${b0}, theta_c: ${theta_c}, T: ${T}"
  TwoPole(b0: b0, a1: a1, a2: a2, y_1: 0, y_2: 0)


proc twoPoleSearchPeak*(freq: float, R: float): TwoPole =
  let f = goldenSectionSearch((f: float) => (twoPole(f, R).peakFreq - freq).abs(), 0, SAMPLE_RATE/2)
  twoPole(f, R)


proc plotMeasurement(freqs: seq[float], ampsMeas: seq[float], ampsTheo: seq[float]) =
  var p = createSinglePlot()
  p.plot(freqs, ampsMeas, "-")
  p.plot(freqs, ampsTheo, "-")
  p.show()
  p.run()


proc freqRange(f1 = 20f, f2 = 10000f, steps = 1000): seq[float] =
  echo &"Resolution: {(f2-f1) / steps} hz"
  result = newSeqOfCap[float](steps + 1)
  for i in 0 .. steps:
    let f = f1 + (f2-f1) / steps * i
    result.add(f)


proc measureFilter*[Filter](filter: var Filter) =
  let freqs = freqRange()

  const duration = 1.0

  var dataO = generateSilence(duration)

  var ampsMeas = newSeqOfCap[float](freqs.len)
  var ampsTheo = newSeqOfCap[float](freqs.len)

  var peakF = 0.0
  var peakA = 0.0

  for f in freqs:
    let dataI = generateSine(f, 1.0, duration)
    filter.reset()
    filter.process(dataI, dataO)
    let ampMeas = dataO.meanAbs() * sqrt(2.0)
    let ampTheo = filter.response(f).abs()
    ampsMeas.add(ampMeas)
    ampsTheo.add(ampTheo)
    if ampMeas > peakA:
      peakF = f
      peakA = ampMeas

  echo &"Peak frequency: {peakF} hz (amplitude: {peakA})"
  plotMeasurement(freqs, ampsMeas, ampsTheo)

when isMainModule:

  var filter = twoPoleSearchPeak(440, 0.99)
  echo &"peak frequency: {filter.peakFreq}"
  measureFilter(filter)
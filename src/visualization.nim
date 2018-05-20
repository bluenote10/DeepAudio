import cairo
import lenientops
import sequtils
import sugar
import strformat

import audiotypes
import waveio
import filters
import generator

#[
type
  Bounds* = tuple[minX, maxX, minY, maxY: float]

proc setupScaledSurface*(bounds: Bounds, resX, resY: int): (PSurface, PContext) =

  let w = bounds.maxX - bounds.minX
  let h = bounds.maxY - bounds.minY
  let scaleX = 1.0 / (w / resX.float)
  let scaleY = 1.0 / (h / resY.float)
  let scale = min(scaleX, scaleY) * 0.90

  var s = image_surface_create(FORMAT_ARGB32, resX.int32, resY.int32)
  var cr = create(s)

  # solid background
  cr.rectangle(0.0, 0.0, resX.float, resY.float)
  cr.set_source_rgb(0.9, 0.9, 0.9)
  cr.fill

  #cr.translate(0.0 - minX*scale, resY.float + minY*scale)
  cr.translate(resX / 2, resY.float / 2)
  cr.scale(scale, -scale)
  cr.translate(-bounds.minX - w/2, -bounds.minY - h/2)

  cr.set_line_width(1.5 / scale)

  (s, cr)

proc debugDraw(w: World) =

  var (s, cr) = setupScaledSurface(w.bounds, 300, 300)

  for i, poly in w.cellPolygons:
    if poly.vertices.len > 0:
      cr.new_sub_path

      cr.set_source_rgba(0.9, 0.2, 0.2, 0.4)
      cr.move_to(poly.vertices[0].x, poly.vertices[0].y)
      for i in 1 ..< poly.vertices.len:
        cr.line_to(poly.vertices[i].x, poly.vertices[i].y)
      cr.close_path
      cr.stroke_preserve

      if w.cellFlags[i].contains(cfWater):
        cr.set_source_rgb(0.1, 0.1, 0.8)
      else:
        let elev = w.cellElevations[i]
        cr.set_source_rgb(elev-0.2, 0.8, elev-0.2)
      cr.fill


  discard write_to_png(s, "world.png")
  destroy(cr)
  destroy(s)
]#

proc draw(data: seq[float], fn: string) =
  let resX = 200.int32
  let resY = 600.int32

  var s = image_surface_create(FORMAT_ARGB32, resX, resY)
  var cr = create(s)

  for i, x in data:
    let y1 = resY / data.len * (i)
    let y2 = resY / data.len * (i+1)
    cr.rectangle(0.0, y1, resX.float, y2)
    let color = x
    cr.set_source_rgb(color, color, color)
    cr.fill()

  discard write_to_png(s, fn)
  destroy(cr)
  destroy(s)


proc processEnsemble(audio: AudioChunk, chunkSize=512) =
  let minMidiKey = 21
  let maxMidiKey = 108

  var dataO = generateSilenceLength(audio.len)
  var output = newSeq[seq[float32]](maxMidiKey-minMidiKey+1)

  var ensemble = newSeq[TwoPole]()
  var i = 0
  for key in minMidiKey .. maxMidiKey:
    let f = key.midiKeyToFreq()
    var filter = twoPoleSearchPeak(f, 0.999) # twoPoleSearchPeak(f, 0.999)
    ensemble.add(filter)

    filter.process(audio, dataO)
    #dataO.normalize()
    #output[i, _] = dataO.data
    let rmsChunks = dataO.rmsChunks(chunkSize=chunkSize)
    output[i] = rmsChunks.normalized.data
    i += 1

    echo &"key = {key:5d}    f = {f:10.1f}    max = {rmsChunks.maxAbs:10.3f}    mean = {rmsChunks.mean:10.3f}"

  for i in 0 ..< output[0].len:
    let data = output.map(row => row[i].float)
    draw(data, &"imgs/img_{i:010d}.png")


when isMainModule:
  if false:
    draw(@[0.1, 0.2, 0.3, 0.9, 1.0, 0.5], "test.png")

  if true:
    let data = loadWave("audio/Sierra Hull  Black River (OFFICIAL VIDEO).wav")
    processEnsemble(data, chunkSize=44100 div 30)

  # ffmpeg -r 30 -i imgs/img_%010d.png -i audio/Sierra\ Hull\ \ Black\ River\ \(OFFICIAL\ VIDEO\).wav -c:v libx264 -c:a aac -pix_fmt yuv420p -crf 23 -r 30 -strict -2 -shortest -y video-from-frames.mp4
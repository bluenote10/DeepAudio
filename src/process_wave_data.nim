import parseopt
import strutils

import arraymancer

import audiotypes
import train_cnn


proc main() =
  var files: seq[string] = @[]
  for kind, key, val in getopt():
    case kind
    of cmdArgument:
      files.add(key)
    of cmdLongOption, cmdShortOption, cmdEnd:
      discard
  echo files

  for file in files:
    let waveData = read_npy[SampleType](file)
    echo waveData.shape

    let audio = AudioChunk(data: waveData.toRawSeq())

    let preprocessed = processEnsemble(audio)

    let outfile = file.replace(".npy", "_preprocessed.npy")
    preprocessed.write_npy(outfile)
    echo "written preprocessed data to: ", outfile


main()
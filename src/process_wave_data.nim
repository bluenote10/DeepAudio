import parseopt

import arraymancer

import audiotypes

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


main()
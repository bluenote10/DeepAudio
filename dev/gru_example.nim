import arraymancer

import sets
import tables
import sequtils
import strformat
import sugar

type
  EncodedSeq = seq[float32]
  CharTensor = Tensor[float32]
  CharTensorVar = Variable[CharTensor]

  Encoding[T] = tuple
    map: Table[T, EncodedSeq]
    encodingLength: int

  CharEnc = Encoding[char]
  WordEnc = Encoding[string]


proc getLongestWord(words: seq[string]): int =
  words.map(w => w.len).max

proc getAllChars(words: seq[string]): seq[char] =
  var charset = initSet[char]()
  for word in words:
    for c in word:
      charset.incl(c)
  result = toSeq(charset.items)

proc getRequiredBits(maxId: int): int =
  var x = maxId
  var requiredBits = 0
  while x > 0:
    requiredBits += 1
    x = x shr 1
  echo &"required bits: {requiredBits}"
  return requiredBits

proc getBitRepresentation(id, requiredBits: int): EncodedSeq =
  result = newSeq[float32](requiredBits)
  for i in 0 ..< requiredBits:
    let mask = 1 shl i
    if (id and mask) == mask:
      result[result.len - i - 1] = 1


proc buildEncoding[T](data: seq[T]): Encoding[T] =
  # we start with id 1 so that every valid char has a non-zero bit
  # and the bitset of all-zeros could be used to encode end-of-sequence.
  let requiredBits = getRequiredBits(data.len + 1)
  var id = 1

  var map = initTable[T, EncodedSeq]()
  for x in data:
    let bits = getBitRepresentation(id, requiredBits)
    echo &"{x} -> {id} -> {bits}"
    map[x] = bits
    id += 1

  return (map: map, encodingLength: requiredBits)


proc getTrainingData(words: seq[string], charMap: CharEnc, wordMap: WordEnc): (CharTensor, CharTensor) =
  let longestWord = words.getLongestWord()

  var X = zeros[float32](words.len, longestWord * charMap.encodingLength)
  var Y = zeros[float32](words.len, wordMap.encodingLength)

  for i, word in words:
    Y[i, _] = wordMap.map[word].toTensor.reshape(1, wordMap.encodingLength)
    for j, chr in word:
      X[i, j*charMap.encodingLength ..< (j+1)*charMap.encodingLength] = charMap.map[chr].toTensor.reshape(1, charMap.encodingLength)

  echo X
  echo Y

let words = @[
  "hello",
  "world",
  "hell",
  "worm",
]
let chars = getAllChars(words)

let charEnc = buildEncoding(chars)
let wordEnc = buildEncoding(words)

discard getTrainingData(words, charEnc, wordEnc)
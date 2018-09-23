import typeinfo
import streams
import strformat

import typetraits


proc serialize*[T: SomeNumber](s: Stream, x: T) =
  s.write(x)

proc serialize*(s: Stream, x: string) =
  # For short strings storing an 8 byte length is
  # quite an overhead. Could be optimized.
  s.write(x.len)
  s.write(x)

proc serialize*[T](s: Stream, x: openarray[T]) =
  s.write(x.len)
  for i in 0 ..< x.len:
    s.serialize(x[i])
  # TODO: we could specialize for primitive types:
  # s.writeData(unsafeAddr(x[0]), sizeof(T) * x.len)

proc serialize*[T: object|tuple](s: Stream, x: T) =
  for field, value in x.fieldPairs:
    s.serialize(value)

proc serialize*[T: ref object|ref tuple](s: Stream, x: T) =
  s.serialize(x[])


proc newEIO(msg: string): ref IOError =
  new(result)
  result.msg = msg

proc read[T](s: Stream, result: var T) {.inline.} =
  ## generic read procedure. Reads `result` from the stream `s`.
  if readData(s, addr(result), sizeof(T)) != sizeof(T):
    raise newEIO("cannot read from stream")


proc deserialize*(s: Stream, T: typedesc[SomeNumber]): T =
  read[T](s, result)

proc deserialize*(s: Stream, T: typedesc[string]): string =
  var len: int
  read[int](s, len)
  result = s.readStr(len)

proc deserialize*[U](s: Stream, T: typedesc[seq[U]]): seq[U] =
  var len: int
  read[int](s, len)
  result = newSeq[U](len)
  for i in 0 ..< len:
    #read[U](s, result[i])
    result[i] = s.deserialize(U)
  # TODO: we could specialize for primitive types:
  # if readData(s, addr(result[0]), sizeof(U) * len) != sizeof(U) * len:
  #   raise newEIO("cannot read from stream")

proc deserialize*[N, U](s: Stream, T: typedesc[array[N, U]]): array[N, U] =
  var len: int
  read[int](s, len)
  doAssert len == result.len
  for i in 0 ..< len:
    #read[U](s, result[i])
    result[i] = s.deserialize(U)
  # TODO: we could specialize for primitive types:
  # if readData(s, addr(result[0]), sizeof(U) * len) != sizeof(U) * len:
  #   raise newEIO("cannot read from stream")

proc deserialize*(s: Stream, T: typedesc[object|tuple]): T =
  for field, value in result.fieldPairs:
    #read[value.type](s, value)
    value = s.deserialize(value.type)

proc deserialize*(s: Stream, T: typedesc[ref object|ref tuple]): T =
  result = T.new
  for field, value in result[].fieldPairs:
    #read[value.type](s, value)
    value = s.deserialize(value.type)

# -----------------------------------------------------------------------------
# High level API
# -----------------------------------------------------------------------------

# TODO: Add option to enable compression

proc storeAsString*[T](x: T): string =
  let stream = newStringStream()
  stream.serialize(x)
  result = stream.data
  stream.close()

proc storeAsFile*[T](x: T, fn: string) =
  let stream = newFileStream(fn, fmWrite)
  stream.serialize(x)
  stream.close()

proc restoreFromString*(s: string, T: typedesc): T =
  let stream = newStringStream(s)
  result = stream.deserialize(T)

proc restoreFromFile*(fn: string, T: typedesc): T =
  let stream = newFileStream(fn)
  result = stream.deserialize(T)

# -----------------------------------------------------------------------------
# Quick checks
# -----------------------------------------------------------------------------

when isMainModule:

  block:
    type
      TestObject = object
        a, b, c: int
      TestObjectRef = ref TestObject

    proc `===`(a, b: TestObjectRef): bool = a[] == b[]

    var s = newStringStream()
    s.serialize(42)
    s.serialize(1.0)
    s.serialize("hey")
    s.serialize(@[1, 2, 3])
    s.serialize([1, 2, 3])
    s.serialize(TestObject(a: 1, b: 2, c: 3))
    s.serialize(TestObjectRef(a: 1, b: 2, c: 3))

    echo "------------------"
    echo "Stream content raw:"
    echo s.data
    echo "Stream content ordinals:"
    for i, c in s.data:
      stdout.write(&"{ord(c):3d} ")
      if i mod 8 == 7:
        stdout.write("\n")
    stdout.write("\n")
    echo "------------------"

    # reset the stream for reading
    s.setPosition(0)
    doAssert s.deserialize(int) == 42
    doAssert s.deserialize(float) == 1.0
    doAssert s.deserialize(string) == "hey"
    doAssert s.deserialize(seq[int]) == @[1, 2, 3]
    doAssert s.deserialize(array[3, int]) == [1, 2, 3]
    doAssert s.deserialize(TestObject) == TestObject(a: 1, b: 2, c: 3)
    doAssert s.deserialize(TestObjectRef) === TestObjectRef(a: 1, b: 2, c: 3)

  import arraymancer
  import sequtils

  block:
    let tensor = toSeq(1 .. 24).toTensor().reshape(2, 3, 4)
    var s = newStringStream()
    s.serialize(tensor)

    s.setPosition(0)
    doAssert s.deserialize(Tensor[int]) == tensor

  block:
    let tensor = toSeq(1 .. 24).toTensor().reshape(1, 24)
    var s = newStringStream()
    s.serialize(tensor)

    s.setPosition(0)
    doAssert s.deserialize(Tensor[int]) == tensor
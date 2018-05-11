import arraymancer

type
  TensorT = Tensor[float]
  Dataset = tuple[x_train: Variable[TensorT], y_train: TensorT]

let ctx = newContext TensorT

proc train(data: Dataset) =
  let n = 32
  let offset = 0
  let x = data.x_train[offset ..< offset + n, _]

var data: Dataset
train(data)

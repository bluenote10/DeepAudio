import sugar

import arraymancer
import audiotypes

type
  TensorT* = Tensor[SampleType]
  VariableT* = Variable[TensorT]

  Dataset* = tuple[X: TensorT, Y: TensorT]
  DatasetGen* = (i: int) -> Dataset
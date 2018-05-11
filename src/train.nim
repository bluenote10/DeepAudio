import generator
import arraymancer
import strformat

type
  TensorT = Tensor[SampleType]

  Dataset = tuple[x_train: Variable[TensorT], y_train: TensorT]

let ctx = newContext TensorT

proc loadData(): Dataset =
  let (audio_train, y_train) = generateRandomNotes(5.0, 100)

  let kernelSize = 100

  #let x_train = audio_train.data.toTensor.unsqueeze(1).unsqueeze(1)
  let audioLength = audio_train.len
  let x_train = audio_train.data.toTensor.reshape(int(audioLength / kernelSize), 1, 1, kernelSize)

  echo x_train.shape
  echo x_train.rank
  echo y_train.shape
  echo y_train.rank
  return (x_train: ctx.variable(x_train), y_train: y_train)


proc train(data: Dataset) =

  let numClasses = data.y_train.shape[0]
  echo &"number of classes: {numClasses}"

  # Config (API is not finished)
  let
    # We randomly initialize all weights and bias between [-0.5, 0.5]
    # In the future requires_grad will be automatically set for neural network layers

    cv1_w = ctx.variable(
      randomTensor(20, 1, 1, 50, 1'f32) .- 0.5'f32,    # Weight of 1st convolution
      requires_grad = true
    )
    cv1_b = ctx.variable(
      randomTensor(20, 1, 1, 1'f32) .- 0.5'f32,       # Bias of 1st convolution
      requires_grad = true
    )

    fc = ctx.variable(
      randomTensor(200, 500, 1'f32) .- 0.5'f32,       # Fully connected: 500 in, 200 ou
      requires_grad = true
    )

    classifier = ctx.variable(
      randomTensor(numClasses, 200, 1'f32) .- 0.5'f32,        # Fully connected: 200 in, numClasses out
      requires_grad = true
    )

  proc model[T](x: Variable[T]): Variable[T] =
    # The formula of the output size of convolutions and maxpools is:
    #   H_out = (H_in + (2*padding.height) - kernel.height) / stride.height + 1
    #   W_out = (W_in + (2*padding.width) - kernel.width) / stride.width + 1

    let cv1 = x.conv2d(cv1_w, cv1_b).relu()      # Conv1: [N, 1, 28, 28] --> [N, 20, 24, 24]     (kernel: 5, padding: 0, strides: 1)
    #let mp1 = cv1.maxpool2D((1,2), (0,0), (1,2)) # Maxpool1: [N, 20, 24, 24] --> [N, 20, 12, 12] (kernel: 2, padding: 0, strides: 2)

    let f = cv1.flatten # mp1.flatten                          # [N, 50, 4, 4] -> [N, 800]
    let hidden = f.linear(fc).relu              # [N, 800]      -> [N, 500]

    result = hidden.linear(classifier)           # [N, 500]      -> [N, 10]

    echo &"shape of x:      {x.value.shape}"
    echo &"shape of cv1:    {cv1.value.shape}"
    #echo &"shape of mp1:    {mp1.value.shape}"
    echo &"shape of f:      {f.value.shape}"
    echo &"shape of hidden: {hidden.value.shape}"
    echo &"shape of output: {result.value.shape}"
    #[
    [2205, 1, 1, 100]
    4
    [88, 220500]
    2
    number of classes: 88
    shape of x:      [32, 1, 1, 100]
    shape of cv1:    [32, 20, 1, 51]
    shape of mp1:    [32, 20, 1, 25]
    shape of f:      [32, 500]
    shape of hidden: [32, 200]
    shape of output: [32, 88]
    ]#

  # Stochastic Gradient Descent (API will change)
  let optim = newSGD[float32](
    cv1_w, cv1_b, fc, classifier, 0.01f # 0.01 is the learning rate
  )

  let x_train = data.x_train
  let y_train = data.y_train
  let n = 32

  # Learning loop
  for epoch in 0 ..< 5:
    for batch_id in 0 ..< x_train.value.shape[0] div n: # some at the end may be missing, oh well ...
      # minibatch offset in the Tensor
      let offset = batch_id * n
      let x = x_train[offset ..< offset + n, _]
      let target = y_train[offset ..< offset + n]

      # Running through the network and computing loss
      let clf = x.model
      #[
      let loss = clf.sparse_softmax_cross_entropy(target)

      if batch_id mod 200 == 0:
        # Print status every 200 batches
        echo "Epoch is: " & $epoch
        echo "Batch id: " & $batch_id
        echo "Loss is:  " & $loss.value.data[0]

      # Compute the gradient (i.e. contribution of each parameter to the loss)
      loss.backprop()

      # Correct the weights now that we have the gradient information
      optim.update()
      ]#

    when false:
      # Validation (checking the accuracy/generalization of our model on unseen data)
      ctx.no_grad_mode:
        echo "\nEpoch #" & $epoch & " done. Testing accuracy"

        # To avoid using too much memory we will compute accuracy in 10 batches of 1000 images
        # instead of loading 10 000 images at once
        var score = 0.0
        var loss = 0.0
        for i in 0 ..< 10:
          let y_pred = X_test[i*1000 ..< (i+1)*1000, _].model.value.softmax.argmax(axis = 1).indices.squeeze
          score += accuracy_score(y_test[i*1000 ..< (i+1)*1000], y_pred)

          loss += X_test[i*1000 ..< (i+1)*1000, _].model.sparse_softmax_cross_entropy(y_test[i*1000 ..< (i+1)*1000]).value.data[0]
        score /= 10
        loss /= 10
        echo "Accuracy: " & $(score * 100) & "%"
        echo "Loss:     " & $loss
        echo "\n"


let data = loadData()
train(data)
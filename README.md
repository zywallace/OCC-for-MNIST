# OCC-for-MNIST
Autoencoder neural network is an unsupervised learning algorithm, setting the tar- get values to be equal to input. It’s always used to learn a compact representation of the input. Especially, if a sparsity constraint is imposed on the hidden units, we have one variant: sparse autoencoder. One class classification is to predict whether sample is positive or negative with no negative sample in training.
In our experiment, sparse autoencoder is implemented and tuned to perform one class classification on common used handwritten digits dataset MNIST. False pos- itive rate of 16.08% and false negative rate of 15.25% are achieved.
Autoencoder could be used to learn the compressed representation of data and especially if we set the threshold, autoencoder would perform one class classification naturally. But what does the
4
j=1
 Figure 2: visualization of trained autoencoder
sparsity constraint function here? We discover it would force each unit to learn the most significant part of information especially when the number of hidden units is large. It does sound like L1 regularization especially L1 usually leads to sparsity. If the sparsity parameter ρ is set very close to zero, then sparsity would increase, which may also perform like dropout method with dropout rate close to 1.
Nevertheless, dropout layer is kind of artificial induced sparsity in training phase and in evaluation phase, dropout does not always lead to sparsity. As for visualization of weight, we could found each hidden unit does learn the correlation of the data. To be more specific, hidden unit use edges to distinguish the samples ie. the input producing the most significant response is the one that aligns in the same direction as the weight vector.

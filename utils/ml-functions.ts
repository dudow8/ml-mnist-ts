// Activation function: Distortion function to create a non-linear function
export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

// Derivative of the sigmoid function
export const sigmoidPrime = (a: number) => a * (1 - a);

// Activation function: Normalize the output to a probability distribution
export const softmax = (x: number[]) => {
  const max = Math.max(...x);
  const exp = x.map(v => Math.exp(v - max));
  const sum = exp.reduce((acc, v) => acc + v, 0);

  return exp.map(v => v / sum);
};

// A loss function that measures the average of the squares of the errors - Not used for classification networks
export const meanSquaredError = (expected: number[], actual: number[]) => {
  return actual.reduce((acc, v, i) => acc + Math.pow(expected[i] - v, 2), 0) / expected.length;
}

// A loss function that measures the difference between the expected and actual output - Used for classification networks
export const crossEntropy = (expected: number[], actual: number[]) => {
  return -expected.reduce((acc, v, i) => acc + v * Math.log(actual[i]), 0);
}

// One-hot encoding is a way to represent categorical data as a binary vector
export const createOneHotEncoded = (index: number) => {
  return Array.from({ length: 10 }, (_, i) => i === index ? 1 : 0);
}

// Create a randomized weights layer
export const createRandomizedWeightsLayer = (inputSize: number, outputSize: number) => {
  return Array.from({ length: outputSize }, () => Array.from({ length: inputSize }, () => Math.random() * 2 - 1));
};

// Create a randomized bias layer
export const createRandomizedBiasLayer = (outputSize: number) => {
  return Array.from({ length: outputSize }, () => Math.random() * 2 - 1);
};

// Create a randomized weights and bias layer
export const createNetworkLayer = (inputSize: number, outputSize: number) => {
  const weights = createRandomizedWeightsLayer(inputSize, outputSize);
  const bias = createRandomizedBiasLayer(outputSize);

  return {
    neurons: Array.from({ length: outputSize }, (_, i) => ({
      weights: weights[i],
      bias: bias[i],
    })),
  };
};

export const shuffleDatasetIndexes = (size: number) => {
  return Array.from({ length: size }, (_, i) => i).sort(() => Math.random() - 0.5);
};
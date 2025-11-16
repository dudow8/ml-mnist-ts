import { ForwardCache, Gradients, Model, Sample } from "../../types";

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

// A loss function that measures the difference between the expected and actual output. Used for classification networks
export const crossEntropy = (expected: number[], actual: number[]) => {
  return -expected.reduce((acc, v, i) => acc + v * Math.log(actual[i]), 0);
}

// One-hot encoding to represent categorical data as a binary vector
export const createOneHotEncoded = (index: number) => {
  return Array.from({ length: 10 }, (_, i) => i === index ? 1 : 0);
}

export const shuffleDatasetIndexes = (size: number) => {
  return Array.from({ length: size }, (_, i) => i).sort(() => Math.random() - 0.5);
};

// Forward pass to calculate the activations and do predictions
export const forward = (model: Model, input: number[]): ForwardCache => {
  const as: number[][] = [input, ...model.layers.map((layer) => Array.from({ length: layer.neurons.length }, () => 0))];
  
  const layers = model.layers;

  layers.forEach((layer, l) => {
    
    // Activations/inputs for the current layer
    const activations = as[l];
    
    layer.neurons.forEach((neuron, n) => {
      const { weights, bias } = neuron;
      // dot product of the weights and the activation plus the bias. These are the logits (z).
      const z = bias + weights.reduce((sum, weight, i) => sum += weight * activations[i], 0);
      // last layer is the output layer, so we don't apply the sigmoid function
      const a = l === layers.length - 1 ? z : sigmoid(z);
      // a^0 = input, so we shift the index by 1. These are the activations for the next layer.
      as[l + 1][n] = a; 
    });
  });

  return { as };
}

// Backward pass to calculate the gradients
export const backward = (model: Model, sample: Sample): Gradients => {

  // initialize the gradients with zeros
  const dws: number[][][] = model.layers.map((layer) => layer.neurons.map((neuron) => neuron.weights.map(() => 0)));
  // initialize the biases with zeros
  const dbs: number[][] = model.layers.map((layer) => layer.neurons.map(() => 0));

  const { input, label } = sample;
  const { as } = forward(model, input);

  const y = createOneHotEncoded(label);
  const logits = as[as.length - 1];
  const classification = softmax(logits);
  const layers = model.layers;

  let dc_dz_next: number[] = [];

  for (let l = layers.length - 1; l >= 0; l--) {

    const input = as[l];
    const layer = layers[l];
  
    // last layer
    if (l === layers.length - 1) {
      const dc_dz = classification.map((probability, i) => probability - y[i]); // δ^L = a^L - y
      
      for (let n = 0; n < layer.neurons.length; n++) {
        const dc_db = dc_dz[n]; // 1 * δ^L
        const dc_dw = input.map((activation) => dc_dz[n] * activation) ; // δ^L * a^(l-1)

        dbs[l][n] = dc_db;
        dws[l][n] = dc_dw;
      };

      dc_dz_next = dc_dz;
    }
    
    // hidden layers
    if (l < layers.length - 1) {
      const dc_dz: number[] = [];
      const output = as[l + 1];
      const next_layer = layers[l + 1];

      for (let n = 0; n < layer.neurons.length; n++) {
        // (W^(l+1)^T * δ^(l+1)) * σ'(a^(l-1))
        dc_dz[n] = next_layer.neurons.reduce((sum, { weights }, nn) => sum += weights[n] * dc_dz_next[nn], 0) * sigmoidPrime(output[n]);

        const dc_db = dc_dz[n]; // 1 * δ^l
        const dc_dw = input.map((activation) => dc_dz[n] * activation) ; // δ^L * a^(l-1)

        dbs[l][n] = dc_db;
        dws[l][n] = dc_dw;
      };

      dc_dz_next = dc_dz;
    }
  };

  const loss = crossEntropy(y, classification);
  
  return { dws, dbs, loss };
}
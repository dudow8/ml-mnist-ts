import {
  MNISTStream,
  createNetworkLayer,
  createOneHotEncoded,
  crossEntropy,
  sigmoid,
  sigmoidPrime,
  softmax,
} from "./utils";

type Neuron = {
  weights: number[];
  bias: number;
};

type Layer = {
  neurons: Neuron[];
};

type Model = {
  layers: Layer[];
  loss?: number;
};

type ForwardCache = {
  as: number[][]; // [layer][neuron]
};

type Gradients = {
  dws: number[][][];  // [layer][neuron][input]
  dbs: number[][];    // [layer][neuron]
  loss: number;
};

type Sample = {
  input: number[];
  label: number;
};

// This is the trained model TODO: create a function to read the model from a file or create a new model
const model: Model = {
  layers: [
    createNetworkLayer(28 * 28, 16),
    createNetworkLayer(16, 16),
    createNetworkLayer(16, 10),
  ],
};

const forward = (model: Model, input: number[]): ForwardCache => {
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

const backward = (model: Model, sample: Sample): Gradients => {

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

type TrainOptions = {
  model: Model;
  learningRate?: number;
};

type TrainResult = {
  model: Model;
  loss: number;
};

const train = async ({ model, learningRate = 0.01 }: TrainOptions) : Promise<TrainResult> =>
  new Promise((resolve) => new MNISTStream("train").using(async (mnist) => {
    const firstRow = await mnist.readAt(0);

    if (!firstRow) {
      throw new Error("No first row");
    }

    // Forward
    const sample: Sample = {
      input: firstRow.pixels,
      label: firstRow.label,
    };

    const { dws, dbs, loss } = backward(model, sample);

    model.layers.forEach((layer, l) => {
      layer.neurons.forEach((neuron, n) => {
        neuron.bias -= learningRate * dbs[l][n];
        neuron.weights = neuron.weights.map((weight, w) => weight - learningRate * dws[l][n][w]);
      });
    });

    resolve({ model, loss });
  })
);


// const test = () => new MNISTStream("test").using(async (mnist) => {
//   const firstRow = await mnist.readAt(0);

//   if (!firstRow) {
//     throw new Error("No first row");
//   }

//   const output = forward(model, firstRow.pixels).as[firstRow.pixels.length - 1];
//   const classification = softmax(output);

//   const ce = crossEntropy(createOneHotEncoded(firstRow.label), classification);
//   const predicted = classification.indexOf(Math.max(...classification));

//   console.log(`The correct answer should be ${firstRow.label}`);
//   console.log(`The predicted answer is ${predicted}`);
//   console.log(`\n\nClassification:\n${classification.map((v, i) => `${i}: ${v.toFixed(4)}`).join("\n")}`);
//   console.log(`\nCross Entropy: ${ce}`);
// });

train({ model }).then(({ loss }) => {
  // TODO:: Persist the model

  console.log(`Loss: ${loss}`);
});
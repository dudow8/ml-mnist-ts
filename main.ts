import {
  MNISTStream,
  createNetworkLayer,
  createOneHotEncoded,
  crossEntropy,
  shuffleDatasetIndexes,
  sigmoid,
  sigmoidPrime,
  softmax,
} from "./utils";

import {
  Model,
  ForwardCache,
  Gradients,
  Sample,
  TrainOptions,
  TrainResult,
} from "./types";
import { loadModel, saveModel } from "./utils/model/model.utils";

const logger = (debug: boolean = false) => {
  if (debug) {
    return (message: any, type: "log" | "table" = "log") => console[type](message);
  }

  return (_: any) => {};
}

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


const train = async (options: TrainOptions) : Promise<TrainResult> =>
  new Promise((resolve) => new MNISTStream("train").using(async (mnist) => {

    const log = logger(options.debug);
    const datasetLenght = mnist.count();
    const { model, learningRate = 0.01, epochs = 10, batchSize = 10 } = options;

    const epochsLoss: number[] = Array.from({ length: epochs }, () => 0);

    log(`[Training Model]\n`);
    log(`Dataset Length: ${datasetLenght}`);
    log(`Epochs: ${epochs}`);
    log(`Batch Size: ${batchSize}`);
    log(`Learning Rate: ${learningRate}`);
    log(`--------------------------------`);

    for(let epoch = 0; epoch < epochs; epoch++) {
      if (epoch > 0) log('--------------');
      log(`Starting Epoch ${epoch + 1} of ${epochs}`);

      const datasetRows = shuffleDatasetIndexes(datasetLenght);
      let epochLoss = 0;
      
      for (let batch = 0; batch < datasetLenght / batchSize; batch++) {
        let gradient: Gradients = { dws: [], dbs: [], loss: 0 };
        const batchRows = datasetRows.slice(batch * batchSize, (batch + 1) * batchSize);

        for(let r of batchRows) {
          const row = await mnist.readAt(r);

          if (!row) {
            throw new Error(`No sample was found for row ${r}`);
          }

          const sample: Sample = { input: row.pixels, label: row.label };
          const { dws, dbs, loss } = backward(model, sample);

          if (!(gradient.dws.length && gradient.dbs.length)) {
            gradient = { dws, dbs, loss };
            continue;
          }

          model.layers.forEach((layer, l) => layer.neurons.forEach((neuron, n) => {
            gradient.dws[l][n] = neuron.weights.map((_, w) => dws[l][n][w] + gradient.dws[l][n][w]);
            gradient.dbs[l][n] += dbs[l][n];
          }));

          gradient.loss += loss;
        };

        const { dws, dbs, loss } = gradient;

        model.layers.forEach((layer, l) => {
          layer.neurons.forEach((neuron, n) => {
            neuron.bias = neuron.bias - learningRate * (dbs[l][n] / batchSize);
            neuron.weights = neuron.weights.map((weight, w) => weight - learningRate * (dws[l][n][w] / batchSize)) ;
          });
        });

        epochLoss += loss;
      };

      epochsLoss[epoch] = epochLoss / datasetLenght;

      log(`Finished Epoch ${epoch + 1} with loss: ${epochsLoss[epoch].toFixed(4)}`);
    }

    log(`--------------------------------`);
    log(`[Finished Training]`);

    resolve({ model, epochsLoss });
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

const trainingParams = {
  model: loadModel(),
  epochs: 10,
  batchSize: 10,
  learningRate: 0.01,
  debug: true
};

train(trainingParams).then(({ model, epochsLoss }) => {
  const log = logger(trainingParams.debug);

  log("\nEpochs Loss:");
  log(epochsLoss.map((loss, i) => ({ 'loss': loss.toFixed(4) })), "table");

  log("Saving model...");
  saveModel(model);
  log("Model saved successfully!");
});
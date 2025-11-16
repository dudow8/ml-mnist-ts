
import { crossEntropy, loadOrCreateModel, log } from "@lib";
import { Gradients, Sample, TrainOptions, TrainResult } from "@types";
import { MNISTStream, saveModel, backward, shuffleDatasetIndexes } from "@lib";
import { DEFAULT_MODEL_CONFIG } from "@const";

export const train = async (options: TrainOptions) : Promise<TrainResult> =>
  new Promise((resolve) => new MNISTStream("train").using(async (mnist) => {
    const startTime = performance.now();

    const datasetLenght = mnist.count();
    const {
      model,
      learningRate = 0.01,
      epochs = 10,
      batchSize = 10,
      activationFunction = DEFAULT_MODEL_CONFIG.activationFunction,
    } = options;

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
          const { dws, dbs, loss } = backward(model, sample, activationFunction);

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

    const executionTime = performance.now() - startTime;
    resolve({ model, epochsLoss, executionTime });
  })
);

// Pretty good configrations for MINIST
const trainingParams: TrainOptions = {
  model: loadOrCreateModel(
    DEFAULT_MODEL_CONFIG.layers, {
      model_name: 'default',
    }
  ),
  epochs: 2,
  batchSize: 64,
  learningRate: 0.01,
};

train(trainingParams).then(({ model, epochsLoss, executionTime }) => {
  log("\nEpochs Loss:");
  log(epochsLoss.map((loss, i) => ({ 'loss': Number(loss.toFixed(4)) })), "table");
  
  const minutes = parseInt((executionTime / 1000 / 60).toString(), 10);
  const seconds = (((executionTime / 1000) % 60)).toFixed(2);

  log(`Executed in ${minutes} minutes and ${seconds} seconds`);

  log("\nSaving model...");
  saveModel(model);
  log("Model saved successfully!");
});


import { DEFAULT_MODEL_CONFIG } from "@const";
import { forward, loadOrCreateModel, log, MNISTStream, softmax } from "@lib";
import { BenchmarkOptions, BenchmarkResult } from "@types";

const benchmark = async (options: BenchmarkOptions) : Promise<BenchmarkResult> => 
  new Promise((resolve) => new MNISTStream("test").using(async (mnist) => {
    const {
      model,
      activationFunction = DEFAULT_MODEL_CONFIG.activationFunction,
    } = options;

    const trainingDatasetLenght = mnist.count();

    log("[Predicting]\n");
    log(`Training Dataset Length: ${trainingDatasetLenght}`);
    log(`--------------------------------`);

    const predictions = {
      samples: trainingDatasetLenght,
      successCount: 0,
      errorCount: 0,
    };

    for (let i = 0; i < trainingDatasetLenght; i++) {
      const sample = await mnist.readAt(i);

      if (!sample) {
        throw new Error(`No sample was found for row ${i}`);
      }

      const { as } = forward(model, sample.pixels, activationFunction);
      const classification = softmax(as[as.length - 1]);
      const predicted = classification.indexOf(Math.max(...classification));
      predictions[sample.label === predicted ? "successCount" : "errorCount"]++;
    }

    resolve(predictions);
}));


const benchmarkParams: BenchmarkOptions = {
  model: loadOrCreateModel(
    DEFAULT_MODEL_CONFIG.layers, {
      model_name: 'default',
    }
  ),
};

benchmark(benchmarkParams).then((predictions) => {
  log("\nBenchmark Results:");
  log({
    'nTotal Samples': predictions.samples,
    'Predicted Successfully': predictions.successCount,
    'Predicted Errorfully': predictions.errorCount,
    'Accuracy': Number((predictions.successCount / predictions.samples).toFixed(4)),
    'Error Rate': Number((predictions.errorCount / predictions.samples).toFixed(4)),
  }, "table");
});

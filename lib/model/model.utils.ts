import path from "path";
import { Model } from "../../types";
import fs from "fs";

const DEFAULT_MODEL_PATH = path.join(process.cwd(), "trained-model.json");

export const createRandomizedWeightsLayer = (inputSize: number, outputSize: number) => {
  return Array.from({ length: outputSize }, () => Array.from({ length: inputSize }, () => Math.random() * 2 - 1));
};

export const createRandomizedBiasLayer = (outputSize: number) => {
  return Array.from({ length: outputSize }, () => Math.random() * 2 - 1);
};

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

export const createModel = (): Model => {
  return {
    layers: [
      createNetworkLayer(28 * 28, 16),
      createNetworkLayer(16, 16),
      createNetworkLayer(16, 10),
    ],
  };
};

export const loadModel = (path = DEFAULT_MODEL_PATH): Model => {
  if (!fs.existsSync(path)) {
    return createModel();
  }

  const model = JSON.parse(fs.readFileSync(path, "utf8")) as Model;
  return model;
};

export const saveModel = (model: Model, path = DEFAULT_MODEL_PATH) => {
  fs.writeFileSync(path, JSON.stringify(model, null, 2), "utf8");
};
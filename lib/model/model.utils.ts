import fs from "fs";
import { ActivationFunction, CreateModelLayer, Model } from "@types";
import { heInit } from "@lib/ml-functions";
import { DEFAULT_MODEL_PATH } from "@const";

export const createRandomizedWeightsLayer = (inputSize: number, outputSize: number, activationFunction: ActivationFunction) => {
  if (activationFunction === 'sigmoid') {
    return Array.from({ length: outputSize }, () => Array.from({ length: inputSize }, () => Math.random() * 2 - 1));
  }

  if (activationFunction === 'relu') {
    return Array.from({ length: outputSize }, () => Array.from({ length: inputSize }, () => heInit(inputSize)));
  }

  throw new Error(`Invalid activation function: ${activationFunction}`);
};

export const createRandomizedBiasLayer = (outputSize: number) => {
  return Array.from({ length: outputSize }, () => Math.random() * 2 - 1);
};

export const createNetworkLayer = (inputSize: number, outputSize: number, activationFunction: ActivationFunction = 'sigmoid') => {
  const weights = createRandomizedWeightsLayer(inputSize, outputSize, activationFunction);
  const bias = createRandomizedBiasLayer(outputSize);

  return {
    neurons: Array.from({ length: outputSize }, (_, i) => ({
      weights: weights[i],
      bias: bias[i],
    })),
  };
};

export const createModel = (layers: CreateModelLayer[], activationFunction: ActivationFunction = 'sigmoid'): Model => {
  return {
    layers: layers.map(({ input, output }) => createNetworkLayer(input, output, activationFunction)),
  };
};

export const loadModel = (path: string): Model | undefined => {
  if (!fs.existsSync(path)) {
    return;
  }

  return JSON.parse(fs.readFileSync(path, "utf8")) as Model;
};

export const loadOrCreateModel = (
  layers: CreateModelLayer[],
  activationFunction: ActivationFunction = 'sigmoid',
  path: string = DEFAULT_MODEL_PATH
): Model => {
  return loadModel(path) || createModel(layers, activationFunction);
};

export const saveModel = (model: Model, path = DEFAULT_MODEL_PATH) => {
  fs.writeFileSync(path, JSON.stringify(model, null, 2), "utf8");
};

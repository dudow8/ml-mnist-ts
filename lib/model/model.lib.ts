
import fs from "fs";
import { ActivationFunction, CreateModelLayer, Model, PersistedModelOptions } from "@types";
import { heInit } from "@lib/ml-functions";
import { DEFAULT_MODEL_CONFIG, DEFAULT_MODEL_NAME, DEFAULT_MODEL_PATH } from "@const";
import path from "path";
import { log } from "@lib/utils";

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
  log(`Initializing new model with ${activationFunction} activation function`);

  return {
    layers: layers.map(({ input, output }) => createNetworkLayer(input, output, activationFunction)),
  };
};

export const loadModel = (path: string): Model | undefined => {
  if (!fs.existsSync(path)) {
    log(`Model not found at ${path}`, "warn");
    return;
  }

  return JSON.parse(fs.readFileSync(path, "utf8")) as Model;
};

const getModelParameters = (options?: PersistedModelOptions) => {
  const {
    activationFunction = DEFAULT_MODEL_CONFIG.activationFunction,
    model_path = DEFAULT_MODEL_PATH,
    model_name = DEFAULT_MODEL_NAME,
  } = options || {};

  return { activationFunction, model_path, model_name };
};

export const loadOrCreateModel = (
  layers: CreateModelLayer[],
  options?: PersistedModelOptions
): Model => {
  const { activationFunction, model_path, model_name } = getModelParameters(options);

  return loadModel(
    path.join(model_path, `${model_name}.${activationFunction}.model.json`)
  ) || createModel(layers, activationFunction);
};

export const saveModel = (
  model: Model,
  options?: PersistedModelOptions
): void => {
  const { activationFunction, model_path, model_name } = getModelParameters(options);

  fs.writeFileSync(
    path.join(model_path, `${model_name}.${activationFunction}.model.json`),
    JSON.stringify(model, null, 2),
    "utf8"
  );
};

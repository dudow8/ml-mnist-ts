
import path from "path";
import { ActivationFunction, CreateModelLayer } from "@types";

export const DEBUGGING = true;

export const DEFAULT_MODEL_NAME = "default";
export const DEFAULT_MODEL_PATH = path.join(process.cwd(), "trained-models");
export const MNIST_DATA_PATH = path.join(process.cwd(), "lib/mnist/data")

export const MODEL_CONFIGS = {
  relu: {
    activationFunction: 'relu',
    layers: [
      { input: 28 * 28, output: 64 },
      { input: 64, output: 32 },
      { input: 32, output: 10 },
    ],
  },
  sigmoid: {
    activationFunction: 'sigmoid',
    layers: [
      { input: 28 * 28, output: 16 },
      { input: 16, output: 16 },
      { input: 16, output: 10 },
    ],
  },
} satisfies Record<ActivationFunction, { activationFunction: ActivationFunction, layers: CreateModelLayer[] }>;

export const DEFAULT_MODEL_CONFIG = MODEL_CONFIGS.relu;

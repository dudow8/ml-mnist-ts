

export type ActivationFunction = 'sigmoid' | 'relu';

export type CreateModelLayer = {
  input: number;
  output: number;
};

export type Neuron = {
  weights: number[];
  bias: number;
};

export type Layer = {
  neurons: Neuron[];
};

export type Model = {
  layers: Layer[];
  loss?: number;
};

export type ForwardCache = {
  as: number[][]; // [layer][neuron]
};

export type Gradients = {
  dws: number[][][];  // [layer][neuron][input]
  dbs: number[][];    // [layer][neuron]
  loss: number;
};

export type Sample = {
  input: number[];
  label: number;
};

export type TrainOptions = {
  model: Model;
  learningRate?: number;
  epochs?: number;
  batchSize?: number;
  activationFunction: ActivationFunction;
  debug?: boolean;
};

export type TrainResult = {
  model: Model;
  epochsLoss: number[];
};

export type BenchmarkOptions = {
  model: Model;
  activationFunction: ActivationFunction;
  debug?: boolean;
};

export type BenchmarkResult = {
  samples: number;
  successCount: number;
  errorCount: number;
};
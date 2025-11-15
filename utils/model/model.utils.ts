import path from "path";
import { Model } from "../../types";
import { createNetworkLayer } from "../ml-functions";
import fs from "fs";

const DEFAULT_MODEL_PATH = path.join(process.cwd(), "trained-model.json");

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
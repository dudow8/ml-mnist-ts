
import { DEBUGGING } from "@const";

const logger = (debug: boolean = false) => {
  if (debug) {
    return (message: any, type: "log" | "table" | "warn" | "error" = "log") => console[type](message);
  }

  return (_: any) => {};
}

export const log = logger(DEBUGGING);

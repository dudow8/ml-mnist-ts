export const logger = (debug: boolean = false) => {
  if (debug) {
    return (message: any, type: "log" | "table" = "log") => console[type](message);
  }

  return (_: any) => {};
}
// mnist-stream.ts
import fs from "fs/promises";
import path from "path";

const TRAIN_PATH = {
  IMAGES: path.join(__dirname, "data", "train-images.idx3-ubyte"),
  LABELS: path.join(__dirname, "data", "train-labels.idx1-ubyte"),
};

const TEST_PATH = {
  IMAGES: path.join(__dirname, "data", "t10k-images.idx3-ubyte"),
  LABELS: path.join(__dirname, "data", "t10k-labels.idx1-ubyte"),
};

export type MNISTRow = {
  label: number;               // 0..9
  pixels: number[];            // length 24*24, values in [0,1], row-major
};

export class MNISTStream {
  private imgFd: fs.FileHandle | null = null;
  private lblFd: fs.FileHandle | null = null;

  private numItems = 0;
  private rows = 0;
  private cols = 0;

  private index = 0; // current cursor

  // Byte offsets where the payload starts (after headers)
  private imgDataStart = 16;
  private lblDataStart = 8;

  private imagesPath: string;  // e.g. ./data/train-images-idx3-ubyte
  private labelsPath: string;   // e.g. ./data/train-labels-idx1-ubyte

  constructor(
    private mode: "train" | "test",
  ) {
    this.imagesPath = this.mode === "train" ? TRAIN_PATH.IMAGES : TEST_PATH.IMAGES;
    this.labelsPath = this.mode === "train" ? TRAIN_PATH.LABELS : TEST_PATH.LABELS;
  }

  /** Open files and read headers (must call before use) */
  async open() {
    this.imgFd = await fs.open(this.imagesPath, "r");
    this.lblFd = await fs.open(this.labelsPath, "r");

    // --- Read image header (16 bytes) ---
    const imgHeader = Buffer.alloc(16);
    await this.imgFd.read(imgHeader, 0, 16, 0);

    const imgMagic = imgHeader.readUInt32BE(0);
    if (imgMagic !== 0x00000803) {
      throw new Error(`Unexpected image magic: ${imgMagic} (expected 2051 / 0x00000803)`);
    }
    this.numItems = imgHeader.readUInt32BE(4);
    this.rows = imgHeader.readUInt32BE(8);
    this.cols = imgHeader.readUInt32BE(12);

    // --- Read labels header (8 bytes) ---
    const lblHeader = Buffer.alloc(8);
    await this.lblFd.read(lblHeader, 0, 8, 0);

    const lblMagic = lblHeader.readUInt32BE(0);
    if (lblMagic !== 0x00000801) {
      throw new Error(`Unexpected label magic: ${lblMagic} (expected 2049 / 0x00000801)`);
    }
    const lblCount = lblHeader.readUInt32BE(4);

    if (lblCount !== this.numItems) {
      throw new Error(`Image count (${this.numItems}) != label count (${lblCount})`);
    }

    this.index = 0;
  }

  /** Total number of rows (items) available */
  count() {
    return this.numItems;
  }

  /** Reset the internal cursor to the beginning */
  reset() {
    this.index = 0;
  }

  /** Close file descriptors */
  async close() {
    await this.imgFd?.close();
    await this.lblFd?.close();
    this.imgFd = null;
    this.lblFd = null;
  }

  /**
   * Read a single row by index (random access).
   * Returns null if index is out of range.
   */
  async readAt(idx: number): Promise<MNISTRow | null> {
    if (!this.imgFd || !this.lblFd) throw new Error("Call open() first");
    if (idx < 0 || idx >= this.numItems) return null;
  
    const pixels = await this.readRawImage(idx); // Uint8Array length 28*28
    const label = await this.readRawLabel(idx);  // number 0–9
  
    // Normalize pixel intensities to [0,1]
    const out = new Array<number>(this.rows * this.cols);
    for (let i = 0; i < pixels.length; i++) {
      out[i] = pixels[i] / 255.0; // 0=black, 1=white
    }
  
    return { label, pixels: out };
  }

  /**
   * Stream-style: returns next row and advances the cursor.
   * Returns null when finished.
   */
  async next(): Promise<MNISTRow | null> {
    if (this.index >= this.numItems) return null;
    const row = await this.readAt(this.index);
    this.index += 1;
    return row;
  }

  public async using<T = void>(callback: (mnist: MNISTStream) => Promise<T>) {
    await this.open();
    await callback(this);
    await this.close();
  }

  // ---------- private helpers ----------

  private async readRawImage(idx: number): Promise<Uint8Array> {
    if (!this.imgFd) throw new Error("images not open");
    const bytesPerImage = this.rows * this.cols; // 28*28 = 784
    const offset = this.imgDataStart + idx * bytesPerImage;

    const buf = Buffer.allocUnsafe(bytesPerImage);
    await this.imgFd.read(buf, 0, bytesPerImage, offset);
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }

  private async readRawLabel(idx: number): Promise<number> {
    if (!this.lblFd) throw new Error("labels not open");
    const offset = this.lblDataStart + idx; // 1 byte per label
    const buf = Buffer.allocUnsafe(1);
    await this.lblFd.read(buf, 0, 1, offset);
    return buf[0];
  }
}
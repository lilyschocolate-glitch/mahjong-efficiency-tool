import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import type { Tile } from './mahjong';

export interface RecognitionResult {
  tile: Tile;
  confidence: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
}

interface ClassificationResult {
  tile: Tile;
  confidence: number;
}

class TileRecognizer {
  private model: cocoSsd.ObjectDetection | null = null;
  private references: Map<Tile, tf.Tensor3D> = new Map();
  private isLoaded = false;

  async init() {
    if (this.isLoaded) return;
    this.model = await cocoSsd.load();
    await this.loadReferences();
    this.isLoaded = true;
  }

  private async loadReferences() {
    const img = new Image();
    img.src = new URL('../assets/tiles.png', import.meta.url).href;
    await new Promise((resolve) => {
      img.onload = resolve;
      img.onerror = () => {
        console.error("Failed to load tile sprite sheet");
        resolve(null);
      };
    });

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d')!;
    canvas.width = 64;
    canvas.height = 64;

    const tileTypes = ['z', 'm', 's', 'p'] as const;
    const tileCounts = [10, 9, 9, 9];

    for (let row = 0; row < 4; row++) {
      const type = tileTypes[row];
      const count = tileCounts[row];
      
      for (let col = 0; col < count; col++) {
        if (row === 0 && col >= 7) continue; 

        const tileName = `${type}${col + 1}` as Tile;
        
        ctx.clearRect(0, 0, 64, 64);
        const sw = img.width / 10;
        const sh = img.height / 4;
        ctx.drawImage(img, col * sw, row * sh, sw, sh, 0, 0, 64, 64);
        
        const tensor = tf.browser.fromPixels(canvas).toFloat().div(tf.scalar(255.0)) as tf.Tensor3D;
        this.references.set(tileName, tensor);
      }
    }
  }

  async recognize(videoElement: HTMLVideoElement): Promise<RecognitionResult[]> {
    if (!this.model) return [];

    const predictions = await this.model.detect(videoElement);
    const results: RecognitionResult[] = [];

    for (const pred of predictions) {
      if (pred.score < 0.3) continue;

      const ratio = pred.bbox[2] / pred.bbox[3];
      if (ratio < 0.4 || ratio > 1.2) continue;

      const [x, y, w, h] = pred.bbox;
      const classified = await this.classify(videoElement, x, y, w, h);
      
      if (classified) {
        results.push({
          tile: classified.tile,
          confidence: classified.confidence,
          bbox: [x, y, w, h] as [number, number, number, number]
        });
      }
    }

    return results;
  }

  private async classify(source: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement, x: number, y: number, w: number, h: number): Promise<ClassificationResult | null> {
    const res = tf.tidy(() => {
      const fullTensor = tf.browser.fromPixels(source);
      
      const ix = Math.max(0, Math.floor(y));
      const iy = Math.max(0, Math.floor(x));
      const iw = Math.min(fullTensor.shape[0] - ix, Math.floor(h));
      const ih = Math.min(fullTensor.shape[1] - iy, Math.floor(w));

      if (iw < 2 || ih < 2) return undefined;

      const cropped = tf.image.cropAndResize(
        fullTensor.expandDims(0) as tf.Tensor4D,
        [[ix / fullTensor.shape[0], iy / fullTensor.shape[1], (ix + iw) / fullTensor.shape[0], (iy + ih) / fullTensor.shape[1]]],
        [0],
        [64, 64]
      ).squeeze([0]).div(tf.scalar(255.0)) as tf.Tensor3D;

      let bestTileCandidate: Tile | null = null;
      let minError = Infinity;

      this.references.forEach((refTensor, tile) => {
        const error = (tf.losses.meanSquaredError(refTensor, cropped) as tf.Tensor).dataSync()[0];
        if (error < minError) {
          minError = error;
          bestTileCandidate = tile;
        }
      });

      const confidence = Math.max(0, 1 - minError);
      
      if (bestTileCandidate && confidence > 0.4) {
        return { tile: bestTileCandidate, confidence };
      }
      return undefined;
    });

    return (res as ClassificationResult) || null;
  }
}

export const recognizer = new TileRecognizer();

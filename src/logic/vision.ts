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

    let predictions = await this.model.detect(videoElement);
    
    // AIモデル（coco-ssd）が牌を1つも見つけられなかった場合のフォールバック
    if (predictions.length === 0 || predictions.every(p => p.score < 0.25)) {
      const fallbackBoxes = await this.detectWhiteBlobs(videoElement);
      predictions = [...predictions, ...fallbackBoxes];
    }

    const results: RecognitionResult[] = [];
    const seenBoxes: [number, number, number, number][] = [];

    for (const pred of predictions) {
      if (pred.score < 0.15) continue;

      const [x, y, w, h] = pred.bbox;
      const ratio = w / h;
      if (ratio < 0.2 || ratio > 1.5) continue;

      // 重複するボックスを避ける（NMS的な簡易処理）
      if (seenBoxes.some(box => this.iou(box, pred.bbox as [number, number, number, number]) > 0.5)) continue;
      seenBoxes.push(pred.bbox as [number, number, number, number]);

      const classified = await this.classify(videoElement, x, y, w, h);
      
      if (classified && classified.confidence > 0.4) {
        results.push({
          tile: classified.tile,
          confidence: classified.confidence,
          bbox: [x, y, w, h] as [number, number, number, number]
        });
      }
    }

    // X座標でソートして左から順に並べる
    return results.sort((a, b) => a.bbox[0] - b.bbox[0]);
  }

  // 「白い牌」らしい領域を単純な画像処理で探す（AIが牌を見逃すのを防ぐ）
  private async detectWhiteBlobs(source: HTMLVideoElement): Promise<any[]> {
    return tf.tidy(() => {
      const tensor = tf.browser.fromPixels(source).resizeBilinear([180, 320]);
      // 白に近い（全チャンネルが一定以上）ピクセルを抽出
      const whiteMask = tensor.min(2).greater(tf.scalar(160)); 
      
      // ここで本来は輪郭抽出をしたいが、性能と実装のシンプルさのため
      // 画像をグリッドで区切り、白いピクセル密度が高いエリアをボックス候補とする
      const boxes: any[] = [];
      const gridH = 10;
      const gridW = 20;
      const cellH = 180 / gridH;
      const cellW = 320 / gridW;
      
      for (let i = 0; i < gridH; i++) {
        for (let j = 0; j < gridW; j++) {
          const slice = whiteMask.slice([i * cellH, j * cellW], [cellH, cellW]);
          const density = slice.mean().dataSync()[0];
          if (density > 0.4) {
            // スケールを元の動画サイズに戻す
            const scaleY = source.videoHeight / 180;
            const scaleX = source.videoWidth / 320;
            boxes.push({
              bbox: [j * cellW * scaleX, i * cellH * scaleY, cellW * scaleX, cellH * scaleY],
              score: 0.2,
              class: 'tile_candidate'
            });
          }
        }
      }
      return boxes;
    });
  }

  private iou(boxA: [number, number, number, number], boxB: [number, number, number, number]): number {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[0] + boxA[2], boxB[0] + boxB[2]);
    const yB = Math.min(boxA[1] + boxA[3], boxB[1] + boxB[3]);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = boxA[2] * boxA[3];
    const boxBArea = boxB[2] * boxB[3];
    return interArea / (boxAArea + boxBArea - interArea);
  }

  private standardize(tensor: tf.Tensor3D): tf.Tensor3D {
    return tf.tidy(() => {
      const { mean, variance } = tf.moments(tensor);
      return tensor.sub(mean).div(tf.sqrt(variance).add(tf.scalar(1e-5)));
    });
  }

  // 画像のエッジ（輪郭）情報を抽出する（Sobelフィルタを手動実装）
  private getEdges(tensor: tf.Tensor3D): tf.Tensor2D {
    return tf.tidy(() => {
      const grayscale = tensor.mean(2).expandDims(-1).expandDims(0) as tf.Tensor4D;
      
      const kernelX = tf.tensor4d([-1, 0, 1, -2, 0, 2, -1, 0, 1], [3, 3, 1, 1]);
      const kernelY = tf.tensor4d([-1, -2, -1, 0, 0, 0, 1, 2, 1], [3, 3, 1, 1]);
      
      const gx = tf.conv2d(grayscale, kernelX, 1, 'same');
      const gy = tf.conv2d(grayscale, kernelY, 1, 'same');
      
      const magnitude = tf.sqrt(tf.add(tf.square(gx), tf.square(gy)));
      return magnitude.squeeze() as tf.Tensor2D;
    });
  }

  private async classify(source: HTMLCanvasElement | HTMLVideoElement | HTMLImageElement, x: number, y: number, w: number, h: number): Promise<ClassificationResult | null> {
    const res = tf.tidy(() => {
      const fullTensor = tf.browser.fromPixels(source);
      
      // 座標の正規化
      const ix = Math.max(0, Math.floor(y));
      const iy = Math.max(0, Math.floor(x));
      const iw = Math.min(fullTensor.shape[0] - ix, Math.floor(h));
      const ih = Math.min(fullTensor.shape[1] - iy, Math.floor(w));

      if (iw < 8 || ih < 8) return undefined;

      const cropped = tf.image.cropAndResize(
        fullTensor.expandDims(0) as tf.Tensor4D,
        [[ix / fullTensor.shape[0], iy / fullTensor.shape[1], (ix + iw) / fullTensor.shape[0], (iy + ih) / fullTensor.shape[1]]],
        [0],
        [64, 64]
      ).squeeze([0]).div(tf.scalar(255.0)) as tf.Tensor3D;

      const standardizedCropped = this.standardize(cropped);
      const edgeCropped = this.getEdges(cropped);

      let bestTileCandidate: Tile | null = null;
      let minCombinedError = Infinity;

      this.references.forEach((refTensor, tile) => {
        const standardizedRef = this.standardize(refTensor);
        const edgeRef = this.getEdges(refTensor);

        // ピクセルベースの誤差（標準化済み）
        const pixelError = (tf.losses.meanSquaredError(standardizedRef, standardizedCropped) as tf.Tensor).dataSync()[0];
        // 輪郭ベースの誤差
        const edgeError = (tf.losses.absoluteDifference(edgeRef, edgeCropped) as tf.Tensor).dataSync()[0];
        
        // 輪郭情報を重視して統合
        const combinedError = (pixelError * 0.3) + (edgeError * 0.7);

        if (combinedError < minCombinedError) {
          minCombinedError = combinedError;
          bestTileCandidate = tile;
        }
      });

      // スコアを算出（1.0に近いほど正確）
      const confidence = Math.max(0, 1 - (minCombinedError / 1.2));
      
      if (bestTileCandidate && confidence > 0.4) {
        return { tile: bestTileCandidate, confidence };
      }
      return undefined;
    });

    return (res as ClassificationResult) || null;
  }
}

export const recognizer = new TileRecognizer();

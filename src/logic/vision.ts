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
  private stackedStandardizedRefs: tf.Tensor4D | null = null;
  private stackedEdgeRefs: tf.Tensor3D | null = null;
  private referenceNames: Tile[] = [];
  private isLoaded = false;
  private lastResults: RecognitionResult[] = []; // 前フレームの認識結果
  private frameCount = 0;

  async init() {
    if (this.isLoaded) return;
    try {
      await tf.setBackend('webgl');
      await tf.ready();
      this.model = await cocoSsd.load();
      await this.loadReferences();
      this.isLoaded = true;
    } catch (e) {
      console.error("Tensorflow init failed, falling back to CPU", e);
      await tf.setBackend('cpu');
      await tf.ready();
      this.model = await cocoSsd.load();
      await this.loadReferences();
      this.isLoaded = true;
    }
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

    const standards: tf.Tensor3D[] = [];
    const edges: tf.Tensor2D[] = [];
    this.referenceNames = [];

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
        
        // ベクトル化のための準備
        const std = this.standardize(tensor);
        const edg = this.getEdges(tensor);
        
        this.referenceNames.push(tileName);
        standards.push(std);
        edges.push(edg);
        
        tensor.dispose();
      }
    }

    // 44枚の参照牌を一つの巨大な塊（スタック）にしてGPUに送る
    this.stackedStandardizedRefs = tf.stack(standards) as tf.Tensor4D;
    this.stackedEdgeRefs = tf.stack(edges) as tf.Tensor3D;
    
    // 中間テンソルの解放（スタック後は不要）
    standards.forEach(t => t.dispose());
    edges.forEach(t => t.dispose());
  }

  async recognize(videoElement: HTMLVideoElement): Promise<RecognitionResult[]> {
    if (!this.model) return [];

    let predictions = await this.model.detect(videoElement);
    
    // AIモデル（coco-ssd）が牌を1つも見つけられなかった場合のフォールバック
    if (predictions.length === 0 || predictions.every((p: any) => p.score < 0.25)) {
      const fallbackBoxes = await this.detectWhiteBlobs(videoElement);
      predictions = [...predictions, ...fallbackBoxes];
    }

    const results: RecognitionResult[] = [];
    const seenBoxes: [number, number, number, number][] = [];

    for (const pred of predictions) {
      if (pred.score < 0.10) continue; 

      const [x, y, w, h] = pred.bbox;
      
      // 巨大枠フィルター (v1.7.3): 画面の半分以上（面積で20%以上または幅/高が60%以上）を占める枠はノイズとして無視
      const screenArea = videoElement.videoWidth * videoElement.videoHeight;
      const boxArea = w * h;
      if (boxArea > screenArea * 0.25 || w > videoElement.videoWidth * 0.6 || h > videoElement.videoHeight * 0.6) continue;
      
      const ratio = w / h;
      if (ratio < 0.2 || ratio > 2.0) continue; 

      if (seenBoxes.some(box => this.iou(box, pred.bbox as [number, number, number, number]) > 0.4)) continue;
      seenBoxes.push(pred.bbox as [number, number, number, number]);

      let classified: ClassificationResult | null = null;
      const cached = this.lastResults.find(prev => this.iou(prev.bbox, pred.bbox as [number, number, number, number]) > 0.7);
      
      if (cached && this.frameCount % 10 !== 0) { // キャッシュ期間を少し延長
        classified = { tile: cached.tile, confidence: cached.confidence };
      } else {
        classified = await this.classify(videoElement, x, y, w, h);
      }
      
      if (classified && classified.confidence > 0.30) {
        results.push({
          tile: classified.tile,
          confidence: classified.confidence,
          bbox: [x, y, w, h] as [number, number, number, number]
        });
      }
    }

    this.frameCount++;
    this.lastResults = results;

    // X座標でソートして左から順に並べる
    return results.sort((a, b) => a.bbox[0] - b.bbox[0]);
  }

    // AIが見逃した「白い矩形（牌）」を、幾何学的な特徴で抽出するバックアップロジック
  private async detectWhiteBlobs(source: HTMLVideoElement): Promise<any[]> {
    const pixels = tf.browser.fromPixels(source);
    const tensor = tf.image.resizeBilinear(pixels, [240, 480]);
    
    // 非同期で平均輝度などを取得
    const avgBrightnessTensor = tensor.mean();
    const avgBrightnessArray = await avgBrightnessTensor.array() as number;

    const threshold = Math.max(120, avgBrightnessArray * 1.3);
    const whiteMask = tensor.min(2).greater(tf.scalar(threshold)); 
    
    const boxes: any[] = [];
    const gridH = 30;
    const gridW = 60;
    const cellH = 240 / gridH;
    const cellW = 480 / gridW;
    
    const scaleY = source.videoHeight / 240;
    const scaleX = source.videoWidth / 480;

    const maskData = await whiteMask.array() as number[][];
    
    for (let i = 0; i < gridH; i++) {
      for (let j = 0; j < gridW; j++) {
        // マスクデータの集計もなるべく効率化（CPUで行うため、全体を先に取得して走査）
        let sum = 0;
        for (let y = i * cellH; y < (i + 1) * cellH; y++) {
          for (let x = j * cellW; x < (j + 1) * cellW; x++) {
            if (maskData[Math.floor(y)][Math.floor(x)]) sum++;
          }
        }
        const density = sum / (cellH * cellW);
        
        if (density > 0.40) {
          boxes.push({
            bbox: [j * cellW * scaleX, i * cellH * scaleY, cellW * scaleX, cellH * scaleY],
            score: 0.15,
            class: 'tile_candidate'
          });
        }
      }
    }
    
    tf.dispose([pixels, tensor, avgBrightnessTensor, whiteMask]);
    return boxes;
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
    if (!this.stackedStandardizedRefs || !this.stackedEdgeRefs) return null;

    const fullTensor = tf.browser.fromPixels(source);
    
    // 座標の正規化
    const ix = Math.max(0, Math.floor(y));
    const iy = Math.max(0, Math.floor(x));
    const iw = Math.min(fullTensor.shape[0] - ix, Math.floor(h));
    const ih = Math.min(fullTensor.shape[1] - iy, Math.floor(w));

    if (iw < 8 || ih < 8) {
      fullTensor.dispose();
      return null;
    }

    const { cropped, standardizedCropped, edgeCropped } = tf.tidy(() => {
      const crp = tf.image.cropAndResize(
        fullTensor.expandDims(0) as tf.Tensor4D,
        [[ix / fullTensor.shape[0], iy / fullTensor.shape[1], (ix + iw) / fullTensor.shape[0], (iy + ih) / fullTensor.shape[1]]],
        [0],
        [64, 64]
      ).squeeze([0]).div(tf.scalar(255.0)) as tf.Tensor3D;
      
      return {
        cropped: crp,
        standardizedCropped: this.standardize(crp),
        edgeCropped: this.getEdges(crp)
      };
    });

    // --- 一括ベクトル判定 (v1.7.1 新ロジック) ---
    // 44枚すべての牌との誤差を同時に計算し、await を一回に激減
    const { bestIdx, confidence } = await (async () => {
      const combinedErrorTensor = tf.tidy(() => {
        // Pixel誤差 (MSE) [44]
        const diffPixel = this.stackedStandardizedRefs!.sub(standardizedCropped.expandDims(0));
        const pixelError = diffPixel.square().mean([1, 2, 3]);

        // Edge誤差 (MAE) [44]
        const diffEdge = this.stackedEdgeRefs!.sub(edgeCropped.expandDims(0));
        const edgeError = diffEdge.abs().mean([1, 2]);

        // 統合誤差（エッジを重視）
        return pixelError.mul(0.2).add(edgeError.mul(0.8));
      });

      const errors = await combinedErrorTensor.array() as number[];
      combinedErrorTensor.dispose();

      let minErr = Infinity;
      let minIdx = -1;
      for (let i = 0; i < errors.length; i++) {
        if (errors[i] < minErr) {
          minErr = errors[i];
          minIdx = i;
        }
      }

      // しきい値を以前より少しだけ緩め（2.2）、モニター越しの判定漏れを防ぐ
      const conf = Math.max(0, 1 - (minErr / 2.2));
      return { bestIdx: minIdx, confidence: conf };
    })();

    let bestTileCandidate: Tile | null = this.referenceNames[bestIdx];

    // 赤ドラ判定
    if (bestTileCandidate && ['m5', 'p5', 's5'].includes(bestTileCandidate as string) && confidence > 0.3) {
      const isRed = await (async () => {
        const factorTensor = tf.tidy(() => {
          const redMask = cropped.slice([10, 10], [44, 44]).unstack(2)[0];
          const blueMask = cropped.slice([10, 10], [44, 44]).unstack(2)[2];
          const greenMask = cropped.slice([10, 10], [44, 44]).unstack(2)[1];
          
          return redMask.greater(tf.scalar(0.45))
            .logicalAnd(redMask.greater(blueMask.mul(tf.scalar(1.2))))
            .logicalAnd(redMask.greater(greenMask.mul(tf.scalar(1.2))))
            .mean();
        });
        const val = await factorTensor.array() as number;
        factorTensor.dispose();
        return val;
      })();

      if (isRed > 0.035) {
        bestTileCandidate = (bestTileCandidate as string).replace('5', '0') as Tile;
      }
    }

    tf.dispose([fullTensor, cropped, standardizedCropped, edgeCropped]);

    // 0.3あれば「吸いつく」ように認識されるように調整
    if (bestTileCandidate && confidence > 0.30) {
      return { tile: bestTileCandidate, confidence };
    }
    return null;
  }
}

export const recognizer = new TileRecognizer();

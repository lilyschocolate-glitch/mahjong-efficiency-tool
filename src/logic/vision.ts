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
  private confidenceThreshold = 0.30;

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

    // --- ROI (Region of Interest) 設定 (v1.8.0) ---
    // ガイド枠に対応する中央領域のみを解析（背景ノイズを100%カット）
    const vw = videoElement.videoWidth;
    const vh = videoElement.videoHeight;
    const roiW = vw * 0.85; // UIのguide-frame 85%に同期
    const roiH = vh * 0.35; // guide-frame 25%より少し広めに
    const roiX = (vw - roiW) / 2;
    const roiY = (vh - roiH) / 2;

    // ROI領域のみを切り出したテンソルでAIに入力
    const roiTensor = tf.tidy(() => {
      const pixels = tf.browser.fromPixels(videoElement);
      const crop = tf.image.cropAndResize(
        pixels.expandDims(0) as tf.Tensor4D,
        [[roiY / vh, roiX / vw, (roiY + roiH) / vh, (roiX + roiW) / vw]],
        [0],
        [roiH, roiW]
      );
      return crop.squeeze([0]).toInt() as tf.Tensor3D;
    });

    const canvas = document.createElement('canvas');
    canvas.width = roiW;
    canvas.height = roiH;
    await tf.browser.toPixels(roiTensor, canvas);
    roiTensor.dispose();

    let predictions = await this.model.detect(canvas as any);
    
    // フォールバック（白牌抽出）もROI内でのみ実行
    if (predictions.length === 0 || predictions.every((p: any) => p.score < 0.20)) {
      const fallbackBoxes = await this.detectWhiteBlobs(canvas as any);
      predictions = [...predictions, ...fallbackBoxes];
    }

    const rawResults: RecognitionResult[] = [];

    for (const pred of predictions) {
      if (pred.score < 0.10) continue; 

      const [x, y, w, h] = pred.bbox;
      const ratio = w / h;
      if (ratio < 0.3 || ratio > 1.8) continue; 
      
      // 牌としての適切なサイズ（ROI内の1/40〜1/4程度）かチェック
      if (w < roiW / 40 || w > roiW / 4) continue;

      const cached = this.lastResults.find(r => this.iou(r.bbox, [x + roiX, y + roiY, w, h]) > 0.7);
      let classified: ClassificationResult | null;
      if (cached && this.frameCount % 10 !== 0) {
        classified = { tile: cached.tile, confidence: cached.confidence };
      } else {
        classified = await this.classify(canvas, x, y, w, h);
      }
      
      if (classified && classified.confidence > this.confidenceThreshold) {
        rawResults.push({
          tile: classified.tile,
          confidence: classified.confidence,
          // 座標を全体画面（video）基準に戻す
          bbox: [x + roiX, y + roiY, w, h] as [number, number, number, number]
        });
      }
    }

    // --- NMS (Non-Maximum Suppression) 実装 (v1.8.0) ---
    // 重なり合った判定のうち、最も「自信がある」牌だけを生き残らせる
    const results = this.applyNMS(rawResults);

    this.frameCount++;
    this.lastResults = results;

    return results.sort((a, b) => a.bbox[0] - b.bbox[0]);
  }

  // 重複判定を除去し、各牌に対して最善の枠のみを残す
  private applyNMS(results: RecognitionResult[]): RecognitionResult[] {
    const sorted = [...results].sort((a, b) => b.confidence - a.confidence);
    const selected: RecognitionResult[] = [];
    const usedIndices = new Set<number>();

    for (let i = 0; i < sorted.length; i++) {
      if (usedIndices.has(i)) continue;

      selected.push(sorted[i]);

      for (let j = i + 1; j < sorted.length; j++) {
        if (usedIndices.has(j)) continue;

        // IOUが高い（重なりが大きい）場合は、片方を消去
        if (this.iou(sorted[i].bbox, sorted[j].bbox) > 0.3) {
          usedIndices.add(j);
        }
      }
    }
    return selected;
  }

    // AIが見逃した「白い矩形（牌）」を、幾何学的な特徴で抽出するバックアップロジック
  private async detectWhiteBlobs(source: HTMLCanvasElement): Promise<any[]> {
    const pixels = tf.browser.fromPixels(source);
    // 高解像度のままROI内を精査
    const tensor = tf.image.resizeBilinear(pixels, [180, 480]);
    
    const avgBrightnessTensor = tensor.mean();
    const avgBrightnessArray = await avgBrightnessTensor.array() as number;

    // 白色の分布を抽出
    const threshold = Math.max(130, avgBrightnessArray * 1.2);
    const whiteMask = tensor.min(2).greater(tf.scalar(threshold)); 
    
    const boxes: any[] = [];
    const gridH = 15;
    const gridW = 40;
    const cellH = 180 / gridH;
    const cellW = 480 / gridW;
    
    const scaleY = source.height / 180;
    const scaleX = source.width / 480;

    const maskData = await whiteMask.array() as number[][];
    const visited = new Array(gridH).fill(0).map(() => new Array(gridW).fill(false));

    // Connected Components Grouping (隣接する白セルを統合)
    for (let i = 0; i < gridH; i++) {
      for (let j = 0; j < gridW; j++) {
        if (maskData[i][j] && !visited[i][j]) {
          // BFSで塊を抽出
          let minR = i, maxR = i, minC = j, maxC = j;
          const queue = [[i, j]];
          visited[i][j] = true;

          while (queue.length > 0) {
            const [r, c] = queue.shift()!;
            minR = Math.min(minR, r); maxR = Math.max(maxR, r);
            minC = Math.min(minC, c); maxC = Math.max(maxC, c);

            [[r-1, c], [r+1, c], [r, c-1], [r, c+1]].forEach(([nr, nc]) => {
              if (nr >= 0 && nr < gridH && nc >= 0 && nc < gridW && maskData[nr][nc] && !visited[nr][nc]) {
                visited[nr][nc] = true;
                queue.push([nr, nc]);
              }
            });
          }

          const w = (maxC - minC + 1) * cellW * scaleX;
          const h = (maxR - minR + 1) * cellH * scaleY;
          
          // 塊が牌のサイズ感にフィットするかチェック
          if (w > source.width / 50 && h > source.height / 20) {
            boxes.push({
              bbox: [minC * cellW * scaleX, minR * cellH * scaleY, w, h],
              score: 0.15,
              class: 'tile_candidate'
            });
          }
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

    // --- 牌種固有の物理的特徴検証 (v1.9.0 Heuristic Check) ---
    const heuristicScore = await this.getHeuristicScore(cropped, bestTileCandidate);
    const finalConfidence = confidence * 0.7 + heuristicScore * 0.3;

    // 赤ドラ判定
    if (bestTileCandidate && ['m5', 'p5', 's5'].includes(bestTileCandidate as string) && finalConfidence > 0.25) {
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
    if (bestTileCandidate && finalConfidence > this.confidenceThreshold) {
      return { tile: bestTileCandidate, confidence: finalConfidence };
    }
    return null;
  }

  // 牌種ごとの物理的な特徴（色・構造）が一致しているかを検証する
  private async getHeuristicScore(cropped: tf.Tensor3D, tile: Tile): Promise<number> {
    const type = tile[0]; // 'm', 's', 'p', 'z'

    const info = await tf.tidy(() => {
      const rgb = cropped.unstack(2);
      const r = rgb[0].mean().arraySync() as number;
      const g = rgb[1].mean().arraySync() as number;
      const b = rgb[2].mean().arraySync() as number;
      
      // 上部（文字・記号エリア）と下部（萬・余白等）のコントラスト
      const topHalf = cropped.slice([0, 0, 0], [32, 64, 3]).mean().arraySync() as number;
      const bottomHalf = cropped.slice([32, 0, 0], [32, 64, 3]).mean().arraySync() as number;
      
      return { r, g, b, topHalf, bottomHalf };
    });

    let score = 0.5; // デフォルトスコア

    if (type === 'm') {
      // 萬子: 赤い要素（漢数字）が上部にあるか、全体的に赤みが強いか
      if (info.r > info.g * 1.1 && info.r > info.b * 1.1) score += 0.3;
      if (info.topHalf < info.bottomHalf) score += 0.2; // 文字部分の密度
    } else if (type === 's') {
      // 索子: 緑色の要素（竹）が支配的か
      if (info.g > info.r * 1.05 && info.g > info.b * 1.05) score += 0.4;
    } else if (type === 'p') {
      // 筒子: 青/緑の円形パターン。平均的な輝度のバランス
      if (Math.abs(info.r - info.b) < 0.1) score += 0.3;
    } else if (type === 'z') {
      // 字牌: 背景の白さが際立っているか
      if (info.r > 0.7 && info.g > 0.7 && info.b > 0.7) score += 0.2;
    }

    return Math.min(1.0, score);
  }
}

export const recognizer = new TileRecognizer();

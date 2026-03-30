import React, { useRef, useState, useCallback, useEffect } from 'react';
import { recognizer } from '../logic/vision';
import type { RecognitionResult } from '../logic/vision';
import { calculateShanten, getAcceptance } from '../logic/mahjong';
import type { Tile } from '../logic/mahjong';
import { calculateScore } from '../logic/scoring';
import type { ScoreResult } from '../logic/scoring';
import MahjongTile from './MahjongTile';

interface Props {
  onDetectedTiles: (tiles: string[]) => void;
  onClose: () => void;
  dora?: Tile[];
}

const CameraCapture: React.FC<Props> = ({ onDetectedTiles, onClose, dora = [] }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RecognitionResult[]>([]);
  const [isInitializing, setIsInitializing] = useState(true);
  const [liveShanten, setLiveShanten] = useState<number | null>(null);
  const [liveAcceptance, setLiveAcceptance] = useState<any[]>([]);
  const [liveScore, setLiveScore] = useState<ScoreResult | null>(null);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsActive(true);
        setError(null);
      }
    } catch (err) {
      console.error("Camera access error:", err);
      setError("カメラにアクセスできませんでした。権限を確認してください。");
    }
  };

  const stopCamera = useCallback(() => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsActive(false);
    }
  }, []);


  const [bufferedTiles, setBufferedTiles] = useState<Tile[]>([]);

  const handleAddToBuffer = () => {
    const newTiles = results.map(r => r.tile as Tile);
    setBufferedTiles(prev => {
      const combined = [...prev, ...newTiles].slice(0, 14);
      return combined.sort((a, b) => {
        if (a[0] !== b[0]) return a[0].localeCompare(b[0]);
        return a[1].localeCompare(b[1]);
      });
    });
  };

  const clearBuffer = () => setBufferedTiles([]);

  const handleConfirm = () => {
    if (bufferedTiles.length > 0) {
      onDetectedTiles(bufferedTiles);
    } else if (results.length > 0) {
      onDetectedTiles(results.map(r => r.tile));
    }
  };

  const handleRemoveBuffered = (idx: number) => {
    setBufferedTiles(prev => prev.filter((_, i) => i !== idx));
  };
  
  // Initialize recognizer
  useEffect(() => {
    const init = async () => {
      try {
        await recognizer.init();
        setIsInitializing(false);
      } catch (e) {
        console.error("Recognizer init failed", e);
      }
    };
    init();
    startCamera();
    return () => stopCamera();
  }, [stopCamera]);

  // Detection Loop
  useEffect(() => {
    let animationId: number;
    const detect = async () => {
      if (videoRef.current && isActive && !isInitializing) {
        try {
          const detections = await recognizer.recognize(videoRef.current);
          setResults(detections);
          drawOverlay(detections);
        } catch (e) {
          console.error("Detection error", e);
        }
      }
      animationId = requestAnimationFrame(detect);
    };
    detect();
    return () => cancelAnimationFrame(animationId);
  }, [isActive, isInitializing]);

  // Live Analysis Logic
  useEffect(() => {
    if (results.length >= 1) {
      const tiles = results.map((r: RecognitionResult) => r.tile as Tile);
      const s = calculateShanten(tiles);
      setLiveShanten(s);

      if (s === -1 && tiles.length === 14) {
        setLiveScore(calculateScore(tiles, dora));
      } else {
        setLiveScore(null);
      }

      if (tiles.length === 13 || tiles.length === 14) {
        const acc = getAcceptance(tiles.slice(0, 13));
        setLiveAcceptance(acc);
      } else {
        setLiveAcceptance([]);
      }
    } else {
      setLiveShanten(null);
      setLiveAcceptance([]);
    }
  }, [results]);

  const drawOverlay = (detections: RecognitionResult[]) => {
    if (!overlayRef.current || !videoRef.current) return;
    const canvas = overlayRef.current;
    const video = videoRef.current;
    
    // Sync size
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const displayRatio = canvas.width / canvas.height;
    const videoRatio = video.videoWidth / video.videoHeight;

    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;

    if (videoRatio > displayRatio) {
      // ビデオの方が横長 -> 左右がクロップされる
      scale = canvas.height / video.videoHeight;
      offsetX = (video.videoWidth * scale - canvas.width) / 2;
    } else {
      // ビデオの方が縦長（または同じ） -> 上下がクロップされる
      scale = canvas.width / video.videoWidth;
      offsetY = (video.videoHeight * scale - canvas.height) / 2;
    }

    ctx.strokeStyle = '#22c55e'; // さわやかな緑色
    ctx.lineWidth = 4;
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 4;
    ctx.shadowColor = 'rgba(0,0,0,0.5)';
    
    ctx.fillStyle = '#22c55e';
    ctx.font = 'bold 14px sans-serif';

    detections.forEach(det => {
      const [x, y, w, h] = det.bbox;
      const sx = x * scale - offsetX;
      const sy = y * scale - offsetY;
      const sw = w * scale;
      const sh = h * scale;

      // 確信度が低いものは半透明にする
      ctx.globalAlpha = det.confidence > 0.6 ? 1.0 : 0.5;

      // 角丸の矩形を描画
      drawRoundedRect(ctx, sx, sy, sw, sh, 4);
      
      // ラベル背景
      const label = `${det.tile}`;
      const metrics = ctx.measureText(label);
      ctx.fillRect(sx, sy - 20, metrics.width + 10, 20);
      
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, sx + 5, sy - 5);
      ctx.fillStyle = '#22c55e';
    });
  };

  const drawRoundedRect = (ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) => {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.stroke();
  };

  return (
    <div className="camera-overlay">
      <div className="camera-container">
        <div className="camera-header">
          <h3>AI 牌認識カメラ</h3>
          <button className="close-btn" onClick={() => { stopCamera(); onClose(); }}>✕</button>
        </div>

        {liveShanten !== null && (
          <div className={`live-analysis-bar ${liveShanten <= 0 ? (liveShanten === -1 ? 'win' : 'tenpai') : ''}`}>
            <div className="live-main-status">
              <span className="live-shanten-text">
                {liveShanten === -1 ? '🀄 和了！' : liveShanten === 0 ? '🔥 聴牌（テンパイ）！' : `${liveShanten}向聴`}
              </span>
            </div>
            {liveAcceptance.length > 0 && liveShanten !== -1 && (
              <div className="live-waits-row">
                <span className="label">{liveShanten <= 0 ? '待ち:' : '有効:'}</span>
                <div className="mini-tiles-scroll">
                  {liveAcceptance.map(a => (
                    <MahjongTile key={a.tile} tile={a.tile} size="small" className="mini-tile" />
                  ))}
                </div>
              </div>
            )}
            {liveScore && liveScore.han > 0 && (
              <div className="live-score-preview">
                <span className="points">{liveScore.pointsCustom || `${liveScore.points}点`}</span>
              </div>
            )}
          </div>
        )}
        
        <div className="video-wrapper">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={overlayRef} className="detection-overlay" />
          
          {/* 撮影ガイド枠 */}
          <div className="camera-guide-frame">
            <div className="guide-corners top-left"></div>
            <div className="guide-corners top-right"></div>
            <div className="guide-corners bottom-left"></div>
            <div className="guide-corners bottom-right"></div>
            <div className="guide-text">ここに手牌を合わせてください</div>
          </div>

          {error && <div className="camera-error">{error}</div>}
          {(isInitializing || !isActive) && !error && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <span>AIモデルを読み込み中...</span>
            </div>
          )}
        </div>

        <div className="camera-controls">
          {/* 累積された牌のプレビュー */}
          <div className="buffered-tiles-preview">
            <div className="section-header">
              <span className="label">スキャン済み: {bufferedTiles.length}/14枚</span>
              {bufferedTiles.length > 0 && <button className="text-btn" onClick={clearBuffer}>クリア</button>}
            </div>
            <div className="mini-tiles-row">
              {bufferedTiles.map((tile, i) => (
                <div key={`${tile}-${i}`} className="mini-tile-wrapper" onClick={() => handleRemoveBuffered(i)}>
                  <MahjongTile tile={tile} size="small" />
                </div>
              ))}
              {[...Array(Math.max(0, 14 - bufferedTiles.length))].map((_, i) => (
                <div key={`empty-${i}`} className="mini-tile-empty" />
              ))}
            </div>
          </div>
          
          <div className="live-detection-tray">
            <div className="detection-scroll-preview">
              {results.length > 0 ? (
                <div className="mini-tiles-row">
                  {results.map((r, i) => (
                    <MahjongTile key={i} tile={r.tile as Tile} size="small" />
                  ))}
                </div>
              ) : (
                <p className="detecting-text">スキャン中...</p>
              )}
            </div>
            <button 
              className="cam-btn add-btn" 
              onClick={handleAddToBuffer}
              disabled={results.length === 0 || bufferedTiles.length >= 14}
            >
              追加
            </button>
          </div>
          
          <div className="action-buttons">
            <button className="cam-btn cancel-btn" onClick={() => { stopCamera(); onClose(); }}>閉じる</button>
            <button 
              className="cam-btn confirm-btn" 
              onClick={handleConfirm}
              disabled={bufferedTiles.length === 0 && results.length === 0}
            >
              {bufferedTiles.length > 0 ? `解析を開始 (${bufferedTiles.length}枚)` : '現在の牌で解析'}
            </button>
          </div>
        </div>
        
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    </div>
  );
};

export default CameraCapture;

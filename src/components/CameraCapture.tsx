import React, { useRef, useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { recognizer } from '../logic/vision';
import type { RecognitionResult } from '../logic/vision';
import { calculateShanten } from '../logic/mahjong';
import type { Tile } from '../logic/mahjong';
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
  const [isInitializing, setIsInitializing] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RecognitionResult[]>([]);
  const [bufferedTiles, setBufferedTiles] = useState<Tile[]>([]);
  const [liveShanten, setLiveShanten] = useState<number | null>(null);
  const [isFlashing, setIsFlashing] = useState(false);
  const [stableCount, setStableCount] = useState(0);
  const [lastDetectedString, setLastDetectedString] = useState("");

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

  const handleAddToBuffer = () => {
    if (results.length > 0) {
      const newTiles = results.map(r => r.tile as Tile);
      const combined = [...bufferedTiles, ...newTiles].slice(0, 14);
      setBufferedTiles(combined.sort((a, b) => {
        if (a[0] !== b[0]) return a[0].localeCompare(b[0]);
        return a[1].localeCompare(b[1]);
      }));
      setIsFlashing(true);
      setTimeout(() => setIsFlashing(false), 300);
    }
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
  
  // Initialize recognizer & camera parallelly
  useEffect(() => {
    // 映像を真っ先に起動する（ユーザーを待たせない）
    startCamera();
    
    const init = async () => {
      try {
        await recognizer.init();
      } catch (e) {
        console.error("Recognizer init failed", e);
        setError("AIモデルの初期化に失敗しました。");
      } finally {
        setIsInitializing(false);
      }
    };
    init();

    return () => stopCamera();
  }, [stopCamera]);

  // Detection Loop
  useEffect(() => {
    let animationId: number;
    const detect = async () => {
      if (videoRef.current && isActive && !isInitializing && !isProcessing) {
        try {
          setIsProcessing(true);
          const detections = await recognizer.recognize(videoRef.current);
          setResults(detections);
          drawOverlay(detections);
        } catch (e) {
          console.error("Detection error", e);
        } finally {
          setIsProcessing(false);
        }
      }
      animationId = requestAnimationFrame(detect);
    };
    detect();
    return () => cancelAnimationFrame(animationId);
  }, [isActive, isInitializing]);

  // Live Analysis & Auto-Add Logic
  useEffect(() => {
    const currentLiveTiles = results.map((r: RecognitionResult) => r.tile as Tile);
    const combinedTiles = [...bufferedTiles, ...currentLiveTiles].slice(0, 14);

    if (combinedTiles.length >= 1) {
      const s = calculateShanten(combinedTiles);
      setLiveShanten(s);
    } else {
      setLiveShanten(null);
    }

    // オートスキャンロジック: 牌が安定して1.5秒程度（約10フレーム）静止していたら自動追加
    const currentString = currentLiveTiles.sort().join(',');
    if (currentLiveTiles.length > 0 && currentString === lastDetectedString) {
      if (stableCount > 8 && bufferedTiles.length < 14) {
        handleAddToBuffer();
        setStableCount(0);
      } else {
        setStableCount(prev => prev + 1);
      }
    } else {
      setLastDetectedString(currentString);
      setStableCount(0);
    }
  }, [results, bufferedTiles, dora]);

  const drawOverlay = (detections: RecognitionResult[]) => {
    if (!overlayRef.current || !videoRef.current) return;
    const canvas = overlayRef.current;
    const video = videoRef.current;
    
    // Sync size
    const rect = video.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 画面(canvas)と動画(native)の比率から、正確なスケールを算出
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;
    
    // object-fit: cover を考慮したスケーリング
    const scale = Math.max(scaleX, scaleY);
    const offsetX = (video.videoWidth * scale - canvas.width) / 2;
    const offsetY = (video.videoHeight * scale - canvas.height) / 2;

    ctx.strokeStyle = '#22c55e'; // さわやかな緑色
    ctx.lineWidth = 6; // さらに太くして視認性をアップ
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 8; // 光彩エフェクト
    ctx.shadowColor = 'rgba(34, 197, 94, 0.6)';
    
    ctx.fillStyle = '#22c55e';
    ctx.font = 'bold 14px sans-serif';

    detections.forEach(det => {
      const [x, y, w, h] = det.bbox;
      const sx = x * scale - offsetX;
      const sy = y * scale - offsetY;
      const sw = w * scale;
      const sh = h * scale;

      // 確信度が低いものは半透明にする
      ctx.globalAlpha = det.confidence > 0.5 ? 1.0 : 0.6;

      // 角丸の矩形を描画
      drawRoundedRect(ctx, sx, sy, sw, sh, 4);
      
      // ラベル背景
      const confidencePercent = Math.round(det.confidence * 100);
      const label = `${det.tile} (${confidencePercent}%)`;
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

  const portalRoot = document.getElementById('camera-portal');
  if (!portalRoot) return null;

  return createPortal(
    <div className={`camera-overlay ${isFlashing ? 'flashing' : ''}`}>
      <div className="camera-container">
        <div className="camera-header">
          <h3>AI 牌認識カメラ</h3>
          <button className="close-btn" onClick={(e) => { 
            e.stopPropagation();
            stopCamera(); 
            onClose(); 
          }}>✕</button>
        </div>

        {/* ライブ認識トレイを上部に配置: 何を読み取っているか一目でわかる */}
        <div className="live-detection-tray-top">
          {results.length > 0 ? (
            <div className="live-tiles-row">
              {results.map((r, i) => (
                <div key={i} className="live-tile-item">
                  <MahjongTile tile={r.tile as Tile} size="small" />
                  <div className="conf-bar" style={{ height: `${r.confidence * 100}%` }}></div>
                </div>
              ))}
            </div>
          ) : (
            <div className={`status-pill ${isInitializing ? 'loading' : 'scanning'}`}>
              {isInitializing ? 'AIモデル準備中...' : `スキャン中... (候補: ${results.length})`}
            </div>
          )}
        </div>

        {liveShanten !== null && (
          <div className={`live-analysis-bar ${liveShanten <= 0 ? (liveShanten === -1 ? 'win' : 'tenpai') : ''}`}>
            <div className="live-main-status">
              <span className="live-shanten-text">
                {liveShanten === -1 ? '🀄 和了！' : liveShanten === 0 ? '🔥 聴牌！' : `${liveShanten}向聴`}
              </span>
              <span className="live-count-badge">{bufferedTiles.length + results.length > 14 ? 14 : bufferedTiles.length + results.length}/14枚</span>
            </div>
          </div>
        )}
        
        <div className="video-wrapper">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={overlayRef} className="detection-overlay" />
          
          {/* AI起動中の控えめなインジケーター */}
          {isInitializing && (
            <div className="minimal-status-loading">
              <div className="pulse-dot"></div>
              AI起動中 (数秒かかる場合があります)
            </div>
          )}

          {/* 14枚揃った時のオーバーレイ */}
          {bufferedTiles.length === 14 && (
            <div className="completion-overlay">
              <div className="completion-card">
                <div className="check-icon">✓</div>
                <h4>14枚スキャン完了！</h4>
                <p>手牌がすべて揃いました。解析を開始しますか？</p>
                <button className="big-confirm-btn" onClick={handleConfirm}>解析結果を見る</button>
                <button className="retry-btn" onClick={clearBuffer}>やり直す</button>
              </div>
            </div>
          )}
          
          <div className="camera-guide-frame">
            <div className="guide-corners top-left"></div>
            <div className="guide-corners top-right"></div>
            <div className="guide-corners bottom-left"></div>
            <div className="guide-corners bottom-right"></div>
            <div className="guide-text">{bufferedTiles.length}/14 枚スキャン済み</div>
          </div>

          {error && <div className="camera-error">{error}</div>}
        </div>

        <div className="camera-controls">
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
    </div>,
    portalRoot
  );
};

export default CameraCapture;

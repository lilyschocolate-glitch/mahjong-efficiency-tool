import React, { useRef, useState, useCallback, useEffect } from 'react';
import { recognizer } from '../logic/vision';
import type { RecognitionResult } from '../logic/vision';
import { calculateShanten, getAcceptance } from '../logic/mahjong';
import type { Tile } from '../logic/mahjong';
import { calculateScore } from '../logic/scoring';
import type { ScoreResult } from '../logic/scoring';
import MahjongTile from './MahjongTile';

interface Props {
  onCapture: (imageSrc: string) => void;
  onDetectedTiles: (tiles: string[]) => void;
  onClose: () => void;
  dora?: Tile[];
}

const CameraCapture: React.FC<Props> = ({ onCapture, onDetectedTiles, onClose, dora = [] }) => {
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

  const capture = () => {
    if (videoRef.current && canvasRef.current) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageSrc = canvas.toDataURL('image/jpeg');
        onCapture(imageSrc);
      }
    }
  };

  const handleAddDetected = () => {
    const tiles = results.map(r => r.tile);
    if (tiles.length > 0) {
      onDetectedTiles(tiles);
    }
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

    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 3;
    ctx.fillStyle = '#00FF00';
    ctx.font = '16px bold sans-serif';

    detections.forEach(det => {
      const [x, y, w, h] = det.bbox;
      const sx = x * scaleX;
      const sy = y * scaleY;
      const sw = w * scaleX;
      const sh = h * scaleY;

      ctx.strokeRect(sx, sy, sw, sh);
      ctx.fillText(`${det.tile} (${Math.round(det.confidence * 100)}%)`, sx, sy > 20 ? sy - 5 : sy + 20);
    });
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
          {error && <div className="camera-error">{error}</div>}
          {(isInitializing || !isActive) && !error && (
            <div className="loading-overlay">
              <div className="spinner"></div>
              <span>AIモデルを読み込み中...</span>
            </div>
          )}
        </div>

        <div className="camera-controls">
          <div className="detection-info">
            {results.length > 0 ? (
              <p>{results.length} 個の牌を検出しました</p>
            ) : (
              <p>牌を探しています...</p>
            )}
          </div>
          <div className="btn-group">
            <button className="capture-btn" onClick={capture} disabled={!isActive}>
              <div className="inner-circle"></div>
            </button>
            {results.length > 0 && (
              <button className="add-tiles-btn" onClick={handleAddDetected}>
                検出した牌を追加
              </button>
            )}
          </div>
          <p className="hint">牌を水平に並べ、明るい場所でかざしてください</p>
        </div>
        
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    </div>
  );
};

export default CameraCapture;

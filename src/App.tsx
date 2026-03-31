import { useState, useMemo, useEffect } from 'react';
import './App.css';
import TilePicker from './components/TilePicker';
import HandDisplay from './components/HandDisplay';
import CameraCapture from './components/CameraCapture';
import MahjongTile from './components/MahjongTile';
import MahjongIcon from './components/MahjongIcon';
import { calculateShanten, getAcceptance } from './logic/mahjong';
import type { Tile, TileAcceptance } from './logic/mahjong';
import { calculateScore } from './logic/scoring';
import type { ScoreResult } from './logic/scoring';

interface WaitDetail extends TileAcceptance {
  score: ScoreResult;
}

interface Recommendation {
  discard: Tile;
  acceptance: TileAcceptance[];
  totalCount: number;
  bestWaitScore: number;
  isTenpai: boolean;
  waits: WaitDetail[];
}

function App() {
  const [hand, setHand] = useState<Tile[]>([]);
  const [showCamera, setShowCamera] = useState(false);
  const [dora, setDora] = useState<Tile[]>([]);
  const [scoreResult, setScoreResult] = useState<ScoreResult | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isParent, setIsParent] = useState(false);
  const [isTsumo, setIsTsumo] = useState(false);

  useEffect(() => {
    if (showCamera) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [showCamera]);

  useEffect(() => {
    if (hand.length === 14 && calculateShanten(hand) === -1) {
      setScoreResult(calculateScore(hand, dora, isTsumo, true, isParent));
    } else {
      setScoreResult(null);
    }
  }, [hand, dora, isTsumo, isParent]);

  const onTileSelect = (tile: Tile) => {
    if (hand.length < 14) {
      const newHand = [...hand, tile].sort((a, b) => {
        if (a[0] !== b[0]) return a[0].localeCompare(b[0]);
        return a[1].localeCompare(b[1]);
      });
      setHand(newHand);
    }
  };

  const handleRemoveTile = (index: number) => {
    setHand(prev => prev.filter((_, i) => i !== index));
  };


  const onDetectedTiles = (newTiles: string[]) => {
    // 認識した牌セットで現在の牌を置き換える
    const sortedTiles = [...(newTiles as Tile[])].sort((a, b) => {
      if (a[0] !== b[0]) return a[0].localeCompare(b[0]);
      return a[1].localeCompare(b[1]);
    });
    setHand(sortedTiles);
    setShowCamera(false);
  };

  const handleDoraClick = (tile: Tile) => {
    if (dora.includes(tile)) {
      setDora(dora.filter(t => t !== tile));
    } else if (dora.length < 5) {
      setDora([...dora, tile]);
    }
  };

  const handleReset = () => {
    setHand([]);
    setDora([]);
    setScoreResult(null);
    setCapturedImage(null);
    setShowCamera(false);
  };

  const shanten = useMemo(() => {
    if (hand.length === 0) return null;
    return calculateShanten(hand);
  }, [hand]);

  // 現時点での待ち詳細（13枚の時用）
  const currentWaits = useMemo((): WaitDetail[] => {
    if (hand.length === 13 && shanten === 0) {
      const accs = getAcceptance(hand);
      return accs.map(acc => ({
        ...acc,
        score: calculateScore([...hand, acc.tile], dora, isTsumo, true, isParent)
      })).sort((a, b) => b.score.points - a.score.points);
    }
    return [];
  }, [hand, shanten, dora, isTsumo, isParent]);

  const recommendations = useMemo((): Recommendation[] => {
    if (hand.length !== 14) return [];
    
    const results: Recommendation[] = [];
    const uniqueTiles = Array.from(new Set(hand));
    
    uniqueTiles.forEach(discardTile => {
      const remainingHand = [...hand];
      const idx = remainingHand.indexOf(discardTile);
      remainingHand.splice(idx, 1);
      
      const currentShanten = calculateShanten(remainingHand);
      const acc = getAcceptance(remainingHand);
      const total = acc.reduce((sum, item) => sum + item.count, 0);
      
      const waits = currentShanten === 0 ? acc.map(a => ({
        ...a,
        score: calculateScore([...remainingHand, a.tile], dora, isTsumo, true, isParent)
      })).sort((a, b) => b.score.points - a.score.points) : [];

      const bestScore = waits.length > 0 ? Math.max(...waits.map(w => w.score.points)) : 0;
      
      results.push({
        discard: discardTile,
        acceptance: acc,
        totalCount: total,
        bestWaitScore: bestScore,
        isTenpai: currentShanten === 0,
        waits: waits
      });
    });
    
    // ソート基準: 1. テンパイ優先 2. 最高点数優先 3. 受け入れ枚数優先
    return results.sort((a, b) => {
      if (a.isTenpai !== b.isTenpai) return a.isTenpai ? -1 : 1;
      if (a.bestWaitScore !== b.bestWaitScore) return b.bestWaitScore - a.bestWaitScore;
      return b.totalCount - a.totalCount;
    });
  }, [hand, dora]);

  const bestRecommendation = recommendations[0];

  return (
    <div className="container">
      <header>
        <h1>🀄️ 多面待ちくん <span className="version-tag">v1.9.1</span></h1>
        <p>手牌をカメラでかざすだけで、待ち牌や牌効率を瞬時に診断します。</p>
      </header>

      <main>
        <section className="input-toggle">
          <button className={`mode-btn ${!showCamera ? 'active' : ''}`} onClick={() => setShowCamera(false)}>
            手動入力
          </button>
          <button className="mode-btn camera-btn-main" onClick={() => setShowCamera(true)}>
            📷 カメラで読み取る
          </button>
        </section>

        {showCamera && (
          <CameraCapture 
            onDetectedTiles={onDetectedTiles}
            onClose={() => setShowCamera(false)} 
            dora={dora}
          />
        )}

        <section className="result-section">
          {/* 最優先: テンパイ状態と待ち */}
          <div className="top-priority-status">
            {hand.length >= 13 && (
              <div className={`status-badge ${shanten === 0 ? 'tenpai-glow' : shanten === -1 ? 'win-glow' : ''}`}>
                {shanten === -1 ? '和了（ホーラ）！' : shanten === 0 ? '聴牌（テンパイ）！' : `${shanten} 向聴`}
              </div>
            )}
            {hand.length < 13 && hand.length > 0 && (
              <div className="status-badge">あと {13 - hand.length} 枚</div>
            )}
          </div>

          {/* 13枚時の待ち牌詳細 */}
          {hand.length === 13 && shanten === 0 && (
            <div className="waits-hero">
              <div className="waits-header">
                <h3>現在の待ち牌</h3>
                <span className="waits-count-total">{currentWaits.reduce((s, w) => s + w.count, 0)}枚</span>
              </div>
              <div className="waits-grid">
                {currentWaits.map(wait => (
                  <div key={wait.tile} className="wait-card">
                    <div className="wait-tile-info">
                      <MahjongTile tile={wait.tile} size="medium" />
                      <span className="wait-count-big">{wait.count}枚</span>
                    </div>
                    <div className="wait-score-info">
                      <div className="wait-points">{wait.score.pointsCustom ? wait.score.pointsCustom : `${wait.score.points}点`}</div>
                      <div className="wait-yaku-tags">
                        {wait.score.yaku.slice(0, 3).map(y => <span key={y} className="mini-yaku">{y}</span>)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 和了時の詳細 */}
          {scoreResult && (
            <div className="win-score-hero">
              <div className="points-display">
                {scoreResult.pointsCustom ? scoreResult.pointsCustom : scoreResult.points} 
                <span className="pts-unit">{scoreResult.pointsCustom ? '' : '点'}</span>
                {scoreResult.label && <span className="score-label-tag">{scoreResult.label}</span>}
              </div>
              <div className="fu-han-display">{scoreResult.fu} 符 {scoreResult.han} 翻</div>
              <div className="yaku-container">
                {scoreResult.yaku.map(y => <span key={y} className="yaku-label">{y}</span>)}
              </div>
            </div>
          )}

          {/* 14枚時の打牌推奨 */}
          {hand.length === 14 && shanten !== -1 && bestRecommendation && (
            <div className="recommendation-hero">
              <div className="rec-title">おすすめの打牌</div>
              <div className="rec-content">
                <div className="rec-tile">
                  <MahjongTile tile={bestRecommendation.discard} size="large" />
                </div>
                <div className="rec-details">
                  {bestRecommendation.isTenpai ? (
                    <div className="tenpai-notice">
                      <span className="text-highlight">テンパイ確定</span>
                      <p>最高打点: {bestRecommendation.bestWaitScore}点</p>
                      <p>待ち枚数: {bestRecommendation.totalCount}枚</p>
                    </div>
                  ) : (
                    <div className="shanten-notice">
                      <p>受け入れ: {bestRecommendation.totalCount}枚</p>
                      <p>次でテンパイの可能性が高い打牌です。</p>
                    </div>
                  )}
                </div>
              </div>
              
              {bestRecommendation.isTenpai && (
                <div className="rec-waits-preview">
                  <p>この牌を切った時の待ち:</p>
                  <div className="mini-waits-list">
                    {bestRecommendation.waits.map(w => (
                      <div key={w.tile} className="mini-wait-item">
                        <MahjongTile tile={w.tile} size="small" />
                        <span>{w.score.pointsCustom ? w.score.pointsCustom : `${w.score.points}点`}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {capturedImage && (
          <section className="preview-section">
            <p>撮影された画像:</p>
            <img src={capturedImage} alt="Captured" className="captured-preview" />
            <div className="preview-controls">
              <button className="clear-btn" onClick={() => setCapturedImage(null)}>削除</button>
            </div>
          </section>
        )}

        <section className="settings-section">
          <div className="setting-item">
            <label>自分</label>
            <div className="toggle-group">
              <button className={`toggle-btn ${!isParent ? 'active' : ''}`} onClick={() => setIsParent(false)}>子</button>
              <button className={`toggle-btn ${isParent ? 'active parent' : ''}`} onClick={() => setIsParent(true)}>親</button>
            </div>
          </div>
          <div className="setting-item">
            <label>アガリ方</label>
            <div className="toggle-group">
              <button className={`toggle-btn ${!isTsumo ? 'active' : ''}`} onClick={() => setIsTsumo(false)}>ロン</button>
              <button className={`toggle-btn ${isTsumo ? 'active tsumo' : ''}`} onClick={() => setIsTsumo(true)}>ツモ</button>
            </div>
          </div>
        </section>

        <section className="hand-section">
          <h2>手牌</h2>
          <HandDisplay hand={hand} onTileRemove={handleRemoveTile} />
          
          <div className="dora-section">
            <h3>🀅 ドラ指定</h3>
            <div className="dora-display">
              {[...Array(5)].map((_, i) => (
                <div key={i} className={`dora-slot ${!dora[i] ? 'empty' : ''}`} onClick={() => dora[i] && handleDoraClick(dora[i])}>
                  {dora[i] ? <MahjongIcon tile={dora[i]} size="medium" /> : <div className="placeholder">+</div>}
                </div>
              ))}
            </div>
            <p className="hint">ドラ牌をクリックすると指定・解除できます</p>
          </div>

          {hand.length > 0 && (
            <button className="clear-btn" onClick={handleReset}>リセット</button>
          )}
        </section>

        <section className="picker-section">
          <h2>牌を選択</h2>
          <TilePicker onTileSelect={onTileSelect} onDoraSelect={handleDoraClick} />
        </section>
      </main>

      <footer>
        <p>© 2026 多面待ちくん - 高精度麻雀AI解析</p>
      </footer>
    </div>
  );
}

export default App;

import { useState } from 'react';
import MahjongTile from './MahjongTile';
import type { Tile } from '../logic/mahjong';

interface Props {
  onTileSelect: (tile: Tile) => void;
  onDoraSelect?: (tile: Tile) => void;
}

const TILE_TYPES = [
  { key: 'm', label: '萬子' },
  { key: 'p', label: '筒子' },
  { key: 's', label: '索子' },
  { key: 'z', label: '字牌' }
];
const TILE_NUMS: Record<string, number> = {
  m: 9, p: 9, s: 9, z: 7
};

const TilePicker: React.FC<Props> = ({ onTileSelect, onDoraSelect }) => {
  const [isDoraMode, setIsDoraMode] = useState(false);

  return (
    <div className="tile-picker">
      <div className="picker-header">
        <button 
          className={`mode-switch ${!isDoraMode ? 'active' : ''}`}
          onClick={() => setIsDoraMode(false)}
        >
          手牌を選択
        </button>
        <button 
          className={`mode-switch dora-mode ${isDoraMode ? 'active' : ''}`}
          onClick={() => setIsDoraMode(true)}
        >
          ドラを選択
        </button>
      </div>

      {TILE_TYPES.map(type => (
        <div key={type.key} className="tile-row">
          <div className="row-label">{type.label}</div>
          <div className="tiles">
            {Array.from({ length: TILE_NUMS[type.key] }, (_, i) => {
              const num = i + 1;
              const tiles = [];
              if (num === 5 && type.key !== 'z') {
                tiles.push(`${type.key}0`);
              }
              tiles.push(`${type.key}${num}`);

              return tiles.map(t => (
                <div key={t} className="picker-tile-wrapper">
                  <MahjongTile
                    tile={t as Tile}
                    size="medium"
                    onClick={() => {
                      if (isDoraMode && onDoraSelect) {
                        onDoraSelect(t as Tile);
                      } else {
                        onTileSelect(t as Tile);
                      }
                    }}
                  />
                  {t[1] === '0' && <span className="red-label">赤</span>}
                </div>
              ));
            })}
          </div>
        </div>
      ))}
    </div>
  );
};

export default TilePicker;

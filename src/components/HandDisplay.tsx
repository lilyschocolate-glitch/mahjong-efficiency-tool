import React from 'react';
import MahjongTile from './MahjongTile';
import type { Tile } from '../logic/mahjong';

interface Props {
  hand: Tile[];
  onTileRemove: (index: number) => void;
}

const HandDisplay: React.FC<Props> = ({ hand, onTileRemove }) => {
  return (
    <div className="hand-display">
      {hand.map((tile, index) => (
        <MahjongTile
          key={`${tile}-${index}`}
          tile={tile}
          size="medium"
          onClick={() => onTileRemove(index)}
        />
      ))}
      {Array.from({ length: 14 - hand.length }, (_, i) => (
        <div key={`empty-${i}`} className="tile tile-empty"></div>
      ))}
    </div>
  );
};

export default HandDisplay;

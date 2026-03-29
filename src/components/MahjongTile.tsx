import React from 'react';
import type { Tile } from '../logic/mahjong';

import MahjongIcon from './MahjongIcon';

interface Props {
  tile: Tile;
  size?: 'small' | 'medium' | 'large';
  onClick?: () => void;
  className?: string;
}

const MahjongTile: React.FC<Props> = ({ tile, size = 'medium', onClick, className = '' }) => {
  return (
    <div 
      className={`mahjong-tile tile-${size} ${className}`} 
      onClick={onClick}
      title={tile}
    >
      <MahjongIcon tile={tile} />
    </div>
  );
};

export default MahjongTile;

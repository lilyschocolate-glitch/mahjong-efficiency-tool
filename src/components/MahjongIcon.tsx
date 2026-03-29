import React from 'react';

interface Props {
  tile: string; // e.g., 'm1', 'z1', 'p9'
  className?: string;
  size?: 'small' | 'medium' | 'large';
}

const MahjongIcon: React.FC<Props> = ({ tile, className, size = 'medium' }) => {
  const type = tile[0];
  const value = parseInt(tile[1]);
  const isRed = (type === 'm' || type === 'p' || type === 's') && value === 0;
  const displayValue = isRed ? 5 : value;
  
  const sizeMap = {
    small: { width: 40, height: 54 },
    medium: { width: 60, height: 81 },
    large: { width: 80, height: 108 }
  };
  
  const { width, height } = sizeMap[size];

  const renderContent = () => {
    switch (type) {
      case 'm': // Manzu
        return renderManzu(displayValue, isRed);
      case 'p': // Pinzu
        return renderPinzu(displayValue, isRed);
      case 's': // Souzu
        return renderSouzu(displayValue, isRed);
      case 'z': // Jihai
        return renderJihai(displayValue);
      default:
        return null;
    }
  };

  const renderManzu = (v: number, red: boolean = false) => {
    const kanji = ['一', '二', '三', '四', '五', '六', '七', '八', '九'];
    return (
      <g>
        <text x="50%" y="42%" textAnchor="middle" dominantBaseline="middle" fontSize="32" fontWeight="bold" fill={red ? "#d32f2f" : "#000"}>
          {kanji[v - 1]}
        </text>
        <text x="50%" y="75%" textAnchor="middle" dominantBaseline="middle" fontSize="24" fontWeight="bold" fill="#d32f2f">
          萬
        </text>
        {red && <circle cx="85" cy="15" r="4" fill="#d32f2f" />}
      </g>
    );
  };

  const renderPinzu = (v: number, red: boolean = false) => {
    const colors = ['#d32f2f', '#1976d2', '#2e7d32'];
    const dots: React.ReactNode[] = [];
    
    // Traditional Mahjong dot style: circle with concentric inner pattern
    const renderSingleDot = (x: number, y: number, r: number, color: string, key: any) => (
      <g key={key}>
        <circle cx={x} cy={y} r={r} fill={color} stroke="#000" strokeWidth="0.5" />
        <circle cx={x} cy={y} r={r * 0.7} fill="none" stroke="white" strokeWidth="0.5" strokeDasharray="1,1" />
        <circle cx={x} cy={y} r={r * 0.3} fill="white" opacity="0.5" />
      </g>
    );

    if (v === 1) {
      dots.push(renderSingleDot(50, 50, 28, colors[0], 'p1'));
    } else {
      const positions = getDotPositions(v);
      positions.forEach((p, i) => {
        let color = red ? colors[0] : (v === 4 ? (i < 2 ? colors[1] : colors[2]) : (v === 6 ? colors[2] : colors[1]));
        if (v === 5 && !red && i === 2) color = colors[0]; // Regular 5p has center red dot
        dots.push(renderSingleDot(p.x, p.y, 8, color, i));
      });
    }
    return <g>
      {dots}
      {red && <circle cx="85" cy="15" r="4" fill="#d32f2f" />}
    </g>;
  };

  const renderSouzu = (v: number, red: boolean = false) => {
    const colors = { green: '#2e7d32', red: '#d32f2f', blue: '#1976d2' };
    
    // Traditional Souzu stick (bamboo) style: segmented rectangle
    const renderStick = (x: number, y: number, color: string, key: any) => (
      <g key={key}>
        <rect x={x - 3} y={y - 12} width="6" height="24" rx="1.5" fill={color} />
        <line x1={x - 3} y1={y} x2={x + 3} y2={y} stroke="rgba(0,0,0,0.2)" strokeWidth="1" />
        <line x1={x - 3} y1={y - 6} x2={x + 3} y2={y - 6} stroke="rgba(0,0,0,0.2)" strokeWidth="1" />
        <line x1={x - 3} y1={y + 6} x2={x + 3} y2={y + 6} stroke="rgba(0,0,0,0.2)" strokeWidth="1" />
      </g>
    );

    if (v === 1) {
      return (
        <g fill={red ? colors.red : colors.green}>
          <path d="M50,25 Q65,25 70,40 Q75,60 55,80 Q45,25 50,25" fill={red ? colors.red : colors.green} />
          <path d="M50,25 Q35,25 30,40 Q25,60 45,80 Q55,25 50,25" fill={red ? "#e53935" : "#219150"} />
          <circle cx="50" cy="35" r="5" fill={red ? "#ff8a80" : colors.red} /> 
          <path d="M50,45 L45,60 L55,60 Z" fill={red ? colors.red : colors.blue} />
          {red && <circle cx="85" cy="15" r="4" fill="#d32f2f" />}
        </g>
      );
    }
    const sticks: React.ReactNode[] = [];
    const positions = getStickPositions(v);
    positions.forEach((p, i) => {
      let color = colors.green;
      if (v === 6 && i < 3) color = colors.red;
      if (v === 8 && (i === 0 || i === 2 || i === 5 || i === 7)) color = colors.red;
      if (red) color = colors.red; // Red 5 all sticks are red
      sticks.push(renderStick(p.x, p.y, color, i));
    });
    return <g>
      {sticks}
      {red && <circle cx="85" cy="15" r="4" fill="#d32f2f" />}
    </g>;
  };

  const renderJihai = (v: number) => {
    const names = ['東', '南', '西', '北', '白', '發', '中'];
    const colors = ['#000', '#000', '#000', '#000', '#777', '#2e7d32', '#d32f2f'];
    
    if (v === 5) { // Haku (White)
      return (
        <g>
          <rect x="25" y="30" width="50" height="70" stroke="#777" strokeWidth="4" fill="none" rx="2" />
          <rect x="32" y="37" width="36" height="56" stroke="#ccc" strokeWidth="1" fill="none" rx="1" />
        </g>
      );
    }
    
    if (v === 6) { // Hatsu (Green)
      return (
        <text x="50%" y="55%" textAnchor="middle" dominantBaseline="middle" fontSize="48" fontWeight="bold" fill={colors[v - 1]} fontFamily="'Noto Sans JP', sans-serif">
          {names[v - 1]}
        </text>
      );
    }

    return (
      <text x="50%" y="55%" textAnchor="middle" dominantBaseline="middle" fontSize="48" fontWeight="bold" fill={colors[v - 1]} fontFamily="'Noto Sans JP', sans-serif">
        {names[v - 1]}
      </text>
    );
  };

  return (
    <svg 
      viewBox="0 0 100 135" 
      width={width}
      height={height}
      className={`mahjong-svg-icon ${className}`}
    >
      {/* 3D-like tile body with shadow and rounded corners */}
      <rect x="4" y="8" width="92" height="123" rx="8" fill="#e0e0e0" />
      <rect x="4" y="4" width="92" height="123" rx="8" fill="#ffffff" stroke="#ddd" strokeWidth="1" />
      <rect x="8" y="8" width="84" height="115" rx="6" fill="#ffffff" stroke="#f0f0f0" strokeWidth="1" />
      {renderContent()}
    </svg>
  );
};

// Precise dot patterns for 1-9
function getDotPositions(v: number) {
  const res: {x: number, y: number}[] = [];
  if (v === 2) return [{x:35, y:35}, {x:65, y:85}];
  if (v === 3) return [{x:25, y:25}, {x:50, y:55}, {x:75, y:85}];
  if (v === 4) return [{x:30, y:35}, {x:70, y:35}, {x:30, y:85}, {x:70, y:85}];
  if (v === 5) return [{x:30, y:35}, {x:70, y:35}, {x:50, y:60}, {x:30, y:85}, {x:70, y:85}];
  if (v === 6) return [{x:30, y:35}, {x:70, y:35}, {x:30, y:60}, {x:70, y:60}, {x:30, y:85}, {x:70, y:85}];
  if (v === 7) return [{x:50, y:25}, {x:30, y:45}, {x:70, y:45}, {x:30, y:75}, {x:70, y:75}, {x:30, y:95}, {x:70, y:95}];
  if (v === 8) return [{x:30, y:25}, {x:70, y:25}, {x:30, y:50}, {x:70, y:50}, {x:30, y:75}, {x:70, y:75}, {x:30, y:100}, {x:70, y:100}];
  if (v === 9) return [{x:25, y:30}, {x:50, y:30}, {x:75, y:30}, {x:25, y:60}, {x:50, y:60}, {x:75, y:60}, {x:25, y:90}, {x:50, y:90}, {x:75, y:90}];
  return res;
}

// Precise stick patterns for 2-9
function getStickPositions(v: number) {
  if (v === 2) return [{x:50, y:35}, {x:50, y:85}];
  if (v === 3) return [{x:50, y:35}, {x:30, y:85}, {x:70, y:85}];
  if (v === 4) return [{x:30, y:35}, {x:70, y:35}, {x:30, y:85}, {x:70, y:85}];
  if (v === 5) return [{x:30, y:35}, {x:70, y:35}, {x:50, y:60}, {x:30, y:85}, {x:70, y:85}];
  if (v === 6) return [{x:30, y:40}, {x:50, y:40}, {x:70, y:40}, {x:30, y:85}, {x:50, y:85}, {x:70, y:85}];
  if (v === 7) return [{x:50, y:30}, {x:30, y:65}, {x:50, y:65}, {x:70, y:65}, {x:30, y:95}, {x:50, y:95}, {x:70, y:95}];
  if (v === 8) return [{x:30, y:35}, {x:50, y:35}, {x:70, y:35}, {x:50, y:65}, {x:30, y:95}, {x:50, y:95}, {x:70, y:95}, {x:50, y:50}];
  if (v === 9) return [{x:30, y:30}, {x:50, y:30}, {x:70, y:30}, {x:30, y:65}, {x:50, y:65}, {x:70, y:65}, {x:30, y:100}, {x:50, y:100}, {x:70, y:100}];
  return [];
}

export default MahjongIcon;

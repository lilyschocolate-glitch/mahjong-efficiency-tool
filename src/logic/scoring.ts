import type { Tile } from './mahjong';

/**
 * 手牌の面子分解結果
 */
export interface HandDecomposition {
  melds: Tile[][];
  head: Tile[];
}

/**
 * 点数計算の結果
 */
export interface ScoreResult {
  han: number;
  fu: number;
  points: number;
  pointsCustom?: string; // "2000-4000" などのツモ時表記用
  yaku: string[];
  isYakuman: boolean;
  label?: string;
}

/**
 * 役判定と点数計算を行う
 */
export function calculateScore(
  hand: Tile[], 
  dora: Tile[] = [], 
  isZumo: boolean = false, 
  isRiichi: boolean = true, 
  isParent: boolean = false
): ScoreResult {
  if (hand.length !== 14) return { han: 0, fu: 0, points: 0, yaku: [], isYakuman: false };

  // 1. 役満の特殊判定
  const yakuman = judgeYakuman(hand);
  if (yakuman.length > 0) {
    const han = yakuman.length * 13;
    const res = calculatePoints(han, 0, isParent, isZumo);
    return { han, fu: 0, points: res.total, pointsCustom: res.custom, yaku: yakuman, isYakuman: true, label: res.label };
  }

  // 2. 七対子の特殊判定
  const chiitoiHan = judgeChiitoi(hand, dora);
  if (chiitoiHan > 0) {
    const yaku = ['七対子'];
    if (isRiichi) yaku.push('立直');
    if (isZumo) yaku.push('門前清自摸和');
    const totalHan = chiitoiHan + (isRiichi ? 1 : 0) + (isZumo ? 1 : 0);
    const res = calculatePoints(totalHan, 25, isParent, isZumo);
    return { han: totalHan, fu: 25, points: res.total, pointsCustom: res.custom, yaku, isYakuman: false, label: res.label };
  }

  // 3. 通常形の判定
  const decompositions = decomposeHand(hand);
  let bestResult: ScoreResult = { han: 0, fu: 0, points: 0, yaku: [], isYakuman: false };

  for (const deco of decompositions) {
    const result = judgeYaku(deco, hand, dora, isZumo, isRiichi, isParent);
    if (result.han > bestResult.han || (result.han === bestResult.han && result.fu > bestResult.fu)) {
      bestResult = result;
    }
  }

  return bestResult;
}

/**
 * 手牌を面子に分解する
 */
function decomposeHand(hand: Tile[]): HandDecomposition[] {
  const counts = new Array(38).fill(0);
  for (const tile of hand) {
    const type = tile[0];
    let num = parseInt(tile[1]);
    if (num === 0) num = 5; // 赤5
    let index = 0;
    if (type === 'm') index = num - 1;
    else if (type === 'p') index = 10 + num - 1;
    else if (type === 's') index = 20 + num - 1;
    else if (type === 'z') index = 30 + num - 1;
    counts[index]++;
  }

  const results: HandDecomposition[] = [];

  function backtrack(index: number, melds: Tile[][], head: Tile[]) {
    if (index >= 38) {
      if (melds.length === 4 && head.length === 2) {
        results.push({ melds: [...melds], head: [...head] });
      }
      return;
    }

    if (counts[index] === 0) {
      backtrack(index + 1, melds, head);
      return;
    }

    // 雀頭
    if (head.length === 0 && counts[index] >= 2) {
      counts[index] -= 2;
      const tile = indexToTile(index);
      backtrack(index, melds, [tile, tile]);
      counts[index] += 2;
    }

    // 刻子
    if (counts[index] >= 3) {
      counts[index] -= 3;
      const tile = indexToTile(index);
      backtrack(index, [...melds, [tile, tile, tile]], head);
      counts[index] += 3;
    }

    // 順子
    if (index < 27 && index % 10 < 7 && counts[index] > 0 && counts[index + 1] > 0 && counts[index + 2] > 0) {
      counts[index]--;
      counts[index + 1]--;
      counts[index + 2]--;
      const t1 = indexToTile(index);
      const t2 = indexToTile(index + 1);
      const t3 = indexToTile(index + 2);
      backtrack(index, [...melds, [t1, t2, t3]], head);
      counts[index]++;
      counts[index + 1]++;
      counts[index + 2]++;
    }
  }

  backtrack(0, [], []);
  return results;
}

function indexToTile(index: number): Tile {
  if (index < 10) return `m${index + 1}`;
  if (index < 20) return `p${index - 10 + 1}`;
  if (index < 30) return `s${index - 20 + 1}`;
  return `z${index - 30 + 1}`;
}

/**
 * 役を判定する
 */
export function judgeYaku(
  deco: HandDecomposition, 
  originalHand: Tile[], 
  dora: Tile[], 
  isZumo: boolean, 
  isRiichi: boolean,
  isParent: boolean
): ScoreResult {
  const yaku: string[] = [];
  let han = isRiichi ? 1 : 0;
  if (isRiichi) yaku.push('立直');
  if (isZumo) {
    yaku.push('門前清自摸和');
    han += 1;
  }
  
  let fu = 20;

  // 1. タンヤオ
  const isTanyao = originalHand.every(t => !isTermino(t));
  if (isTanyao) {
    yaku.push('断幺九');
    han += 1;
  }

  // 2. 役牌
  deco.melds.forEach(m => {
    if (m[0] === m[1] && m[1] === m[2]) {
      const tile = m[0];
      if (tile === 'z5') { yaku.push('役牌:白'); han += 1; }
      if (tile === 'z6') { yaku.push('役牌:發'); han += 1; }
      if (tile === 'z7') { yaku.push('役牌:中'); han += 1; }
    }
  });

  // 3. 平和
  const isAllShunsu = deco.melds.every(m => m[0] !== m[1]);
  const head = deco.head[0];
  const isNotYakuHead = head[0] !== 'z' || (parseInt(head[1]) < 5);
  
  // 待ち判定 (平和のためには両面待ちが必要)
  // 簡易的に判定: ツモアガリの牌が順子の端でないこと
  // 本来はアガリ牌が必要だが、ここでは「平和の可能性がある」として判定
  if (isAllShunsu && isNotYakuHead) {
    yaku.push('平和');
    han += 1;
    fu = isZumo ? 20 : 30; // 平和ロンは30符、ツモは20符
  } else {
    // 符計算の基本
    fu = 20; 
    if (!isZumo) fu += 10; // 門前ロン
    
    // 面子による加符
    deco.melds.forEach(m => {
      const isKotsu = m[0] === m[1];
      if (isKotsu) {
        let add = 2; // 中張牌
        if (isTermino(m[0])) add *= 2; // 幺九牌
        add *= 2; // 暗刻
        fu += add;
      }
    });
    
    // 雀頭による加符
    if (head[0] === 'z' && parseInt(head[1]) >= 5) fu += 2;
    
    // ツモ加符
    if (isZumo && !yaku.includes('平和')) fu += 2;
    
    // 符の切り上げ
    fu = Math.ceil(fu / 10) * 10;
    if (fu < 30) fu = 30;
  }

  // 4. 小三元
  const sangenKotsu = deco.melds.filter(m => m[0][0] === 'z' && parseInt(m[0][1]) >= 5).length;
  const isSangenHead = (head[0] === 'z' && parseInt(head[1]) >= 5);
  if (sangenKotsu === 2 && isSangenHead) {
    yaku.push('小三元');
    han += 2;
  }

  // 4. 一盃口
  const shunsuStrings = deco.melds.filter(m => m[0] !== m[1]).map(m => m.sort().join(''));
  const uniqueShunsu = new Set(shunsuStrings);
  if (shunsuStrings.length - uniqueShunsu.size === 1) {
    yaku.push('一盃口');
    han += 1;
  }

  // 5. 対々和
  const kotsuCount = deco.melds.filter(m => m[0] === m[1]).length;
  if (kotsuCount === 4) {
    yaku.push('対々和');
    han += 2;
    fu = 40;
  }

  // 6. 混一色・清一色
  const suits = new Set(originalHand.filter(t => t[0] !== 'z').map(t => t[0]));
  const hasHonors = originalHand.some(t => t[0] === 'z');
  if (suits.size === 1) {
    if (hasHonors) {
      yaku.push('混一色');
      han += 3;
    } else {
      yaku.push('清一色');
      han += 6;
    }
  }

  // 7. 三色同順
  const sequences = deco.melds.filter(m => m[0] !== m[1]);
  let hasSanshoku = false;
  for (let i = 1; i <= 7; i++) {
    const hasM = sequences.some(m => m.sort().join('') === [`m${i}`, `m${i+1}`, `m${i+2}`].join(''));
    const hasP = sequences.some(m => m.sort().join('') === [`p${i}`, `p${i+1}`, `p${i+2}`].join(''));
    const hasS = sequences.some(m => m.sort().join('') === [`s${i}`, `s${i+1}`, `s${i+2}`].join(''));
    if (hasM && hasP && hasS) hasSanshoku = true;
  }
  if (hasSanshoku) {
    yaku.push('三色同順');
    han += 2;
  }

  // 8. 一気通貫
  let hasItsu = false;
  ['m', 'p', 's'].forEach(suit => {
    const s1 = [`${suit}1`, `${suit}2`, `${suit}3`].join('');
    const s2 = [`${suit}4`, `${suit}5`, `${suit}6`].join('');
    const s3 = [`${suit}7`, `${suit}8`, `${suit}9`].join('');
    const currentSuites = sequences.map(m => m.sort().join(''));
    if (currentSuites.includes(s1) && currentSuites.includes(s2) && currentSuites.includes(s3)) hasItsu = true;
  });
  if (hasItsu) {
    yaku.push('一気通貫');
    han += 2;
  }

  // 9. チャンタ・純チャン
  const allSetsIncludeTerminal = [...deco.melds, deco.head].every(set => set.some(t => isTermino(t)));
  if (allSetsIncludeTerminal && !isTanyao) {
    if (hasHonors) {
      yaku.push('混全帯幺九');
      han += 2;
    } else {
      yaku.push('純全帯幺九');
      han += 3;
    }
  }

  // 10. 三暗刻
  if (kotsuCount >= 3) {
    yaku.push('三暗刻');
    han += 2;
  }

  // ドラ判定
  let doraCount = 0;
  originalHand.forEach(t => {
    // 通常ドラ
    if (dora.includes(t)) doraCount++;
    // 赤ドラ
    if (t[1] === '0') doraCount++;
  });
  
  if (doraCount > 0) {
    yaku.push(`ドラ ${doraCount}`);
    han += doraCount;
  }

  const res = calculatePoints(han, fu, isParent, isZumo);
  return { han, fu, points: res.total, pointsCustom: res.custom, yaku, isYakuman: false, label: res.label };
}

function judgeYakuman(hand: Tile[]): string[] {
  const yaku: string[] = [];
  const counts = new Array(38).fill(0);
  hand.forEach(tile => {
    const type = tile[0];
    let num = parseInt(tile[1]);
    if (num === 0) num = 5;
    let index = 0;
    if (type === 'm') index = num - 1;
    else if (type === 'p') index = 10 + num - 1;
    else if (type === 's') index = 20 + num - 1;
    else if (type === 'z') index = 30 + num - 1;
    counts[index]++;
  });

  // 国士無双 (13面待ちの判定は考慮せず一律役満。13面待ちは14枚目でしか判定できない仕組みのため)
  const terminals = [0, 8, 10, 18, 20, 28, 30, 31, 32, 33, 34, 35, 36];
  if (terminals.every(idx => counts[idx] > 0) && terminals.some(idx => counts[idx] >= 2)) {
    yaku.push('国士無双');
  }

  // 大三元
  if (counts[34] >= 3 && counts[35] >= 3 && counts[36] >= 3) {
    yaku.push('大三元');
  }

  // 字一色
  const ziisou = Array.from({ length: 38 }, (_, i) => i).every(i => counts[i] === 0 || i >= 30);
  if (ziisou) yaku.push('字一色');

  // 緑一色 (2,3,4,6,8s および 發)
  const ryuuisouAllowed = [21, 22, 23, 25, 27, 35]; // 2s, 3s, 4s, 6s, 8s, z6
  const ryuuisou = Array.from({ length: 38 }, (_, i) => i).every(i => counts[i] === 0 || ryuuisouAllowed.includes(i));
  if (ryuuisou) yaku.push('緑一色');

  // 清老頭
  const chinrousoAllowed = [0, 8, 10, 18, 20, 28]; // 1m, 9m, 1p, 9p, 1s, 9s
  const chinrouso = Array.from({ length: 38 }, (_, i) => i).every(i => counts[i] === 0 || chinrousoAllowed.includes(i));
  if (chinrouso) yaku.push('清老頭');

  // 四喜和 (小四喜・大四喜)
  const kazeKotsu = [30, 31, 32, 33].filter(idx => counts[idx] >= 3).length;
  const kazeHead = [30, 31, 32, 33].some(idx => counts[idx] === 2);
  if (kazeKotsu === 4) yaku.push('大四喜');
  else if (kazeKotsu === 3 && kazeHead) yaku.push('小四喜');

  // 九蓮宝燈 (門前限定想定だが入力は14枚)
  ['m', 'p', 's'].forEach((_, sIdx) => {
    const start = sIdx * 10;
    const suitCounts = counts.slice(start, start + 9);
    const sum = suitCounts.reduce((a, b) => a + b, 0);
    if (sum === 14) {
      const churenPattern = [3, 1, 1, 1, 1, 1, 1, 1, 3];
      let isChuren = true;
      for (let i = 0; i < 9; i++) {
        if (suitCounts[i] < churenPattern[i]) isChuren = false;
      }
      if (isChuren) yaku.push('九蓮宝燈');
    }
  });

  // 四暗刻 (全刻子)
  // ここでの判定は簡易的。本来は門前・ロンの区切りが必要。
  // decomposeHand の結果を使用するのが確実だが、judgeYakuman は独立して呼ばれるため
  // 手牌のカウントから判定。
  const kotsuCandidate = Array.from({ length: 38 }, (_, i) => i).filter(i => counts[i] >= 3).length;
  const pairCandidate = Array.from({ length: 38 }, (_, i) => i).filter(i => counts[i] >= 2).length;
  // 4刻子1対子の形
  if (kotsuCandidate === 4 && pairCandidate === 5) { // 刻子4つの時は対子も含まれるため counts>=2 は 5個(4+1) になる
    yaku.push('四暗刻');
  }

  return yaku;
}

function judgeChiitoi(hand: Tile[], dora: Tile[]): number {
  const counts: Record<string, number> = {};
  hand.forEach(t => counts[t] = (counts[t] || 0) + 1);
  const pairs = Object.values(counts).filter(c => c === 2).length;
  if (pairs === 7) {
    let han = 2;
    let doraCount = 0;
    hand.forEach(t => { if (dora.includes(t)) doraCount++; });
    return han + doraCount;
  }
  return 0;
}

export function isTermino(tile: Tile): boolean {
  if (tile[0] === 'z') return true;
  let num = parseInt(tile[1]);
  if (num === 0) num = 5; // 赤5は一九牌ではない
  return num === 1 || num === 9;
}

export interface PointsResult {
  total: number;
  custom?: string; // "2000-4000" 等
  label?: string;
}

/**
 * Mリーグ基準の点数計算
 */
export function calculatePoints(han: number, fu: number, isParent: boolean, isZumo: boolean): PointsResult {
  if (han <= 0) return { total: 0 };
  
  // 満貫以上の判定
  let limitPoints = 0;
  let label = "";

  if (han >= 13) { 
    const multiple = Math.floor(han / 13);
    limitPoints = 8000 * multiple; 
    label = multiple > 1 ? `${multiple}倍役満` : "役満"; 
  }
  else if (han >= 11) { limitPoints = 6000; label = "三倍満"; }
  else if (han >= 8) { limitPoints = 4000; label = "倍満"; }
  else if (han >= 6) { limitPoints = 3000; label = "跳満"; }
  else if (han >= 5 || (han === 4 && fu >= 40) || (han === 3 && fu >= 70)) { limitPoints = 2000; label = "満貫"; }

  if (limitPoints > 0) {
    if (isZumo) {
      if (isParent) {
        const p = limitPoints; // 親の満貫ツモは4000オール
        return { total: p * 3, custom: `${p} ALL`, label };
      } else {
        const pKo = limitPoints / 2;
        const pOya = limitPoints;
        return { total: pKo * 2 + pOya, custom: `${pKo}-${pOya}`, label };
      }
    } else {
      const total = limitPoints * (isParent ? 6 : 4);
      return { total, label };
    }
  }

  // 通常計算 (1-4翻)
  const basic = fu * Math.pow(2, han + 2);
  
  if (isZumo) {
    if (isParent) {
      const p = Math.ceil((basic * 2) / 100) * 100;
      return { total: p * 3, custom: `${p} ALL` };
    } else {
      const pKo = Math.ceil(basic / 100) * 100;
      const pOya = Math.ceil((basic * 2) / 100) * 100;
      return { total: pKo * 2 + pOya, custom: `${pKo}-${pOya}` };
    }
  } else {
    const total = Math.ceil((basic * (isParent ? 6 : 4)) / 100) * 100;
    return { total };
  }
}

/**
 * 麻雀牌を表現する文字列型
 * m: 萬子, p: 筒子, s: 索子, z: 字牌 (1-7: 東南西北白發中)
 */
export type Tile = string;

/**
 * シャンテン数計算の結果
 */
export interface ShantenResult {
  shanten: number;
  bestDiscards: Tile[];
}

/**
 * 手牌からシャンテン数を計算する
 */
export function calculateShanten(hand: Tile[]): number {
  const counts = tileToCounts(hand);
  let minShanten = 8;

  // 標準形 (4面子1雀頭)
  const standardShanten = calculateStandardShanten(counts);
  minShanten = Math.min(minShanten, standardShanten);

  // 七対子
  const chiitoiShanten = calculateChiitoiShanten(counts);
  minShanten = Math.min(minShanten, chiitoiShanten);

  // 国士無双
  const kokushiShanten = calculateKokushiShanten(counts);
  minShanten = Math.min(minShanten, kokushiShanten);

  return minShanten;
}

/**
 * 有効牌の特定
 */
export interface TileAcceptance {
  tile: Tile;
  count: number;
}

export function getAcceptance(hand13: Tile[]): TileAcceptance[] {
  const currentShanten = calculateShanten(hand13);
  const counts = tileToCounts(hand13);
  const acceptance: TileAcceptance[] = [];

  const allTiles: Tile[] = [];
  ['m', 'p', 's'].forEach(t => {
    for (let i = 1; i <= 9; i++) allTiles.push(`${t}${i}`);
  });
  for (let i = 1; i <= 7; i++) allTiles.push(`z${i}`);

  for (const nextTile of allTiles) {
    const nextHand = [...hand13, nextTile];
    if (calculateShanten(nextHand) < currentShanten) {
      const nextCounts = tileToCounts([nextTile]);
      const nextIdx = nextCounts.findIndex(c => c > 0);
      const remaining = 4 - counts[nextIdx];
      if (remaining > 0) {
        acceptance.push({ tile: nextTile, count: remaining });
      }
    }
  }

  return acceptance;
}

/**
 * 補助関数
 */
function tileToCounts(hand: Tile[]): number[] {
  const counts = new Array(38).fill(0);
  for (const tile of hand) {
    const type = tile[0];
    let num = parseInt(tile[1]);
    if (num === 0) num = 5; // 赤5は通常の5として扱う
    let index = 0;
    if (type === 'm') index = num - 1;
    else if (type === 'p') index = 10 + num - 1;
    else if (type === 's') index = 20 + num - 1;
    else if (type === 'z') index = 30 + num - 1;
    counts[index]++;
  }
  return counts;
}

function countMeldsAndTatsu(counts: number[]): { melds: number; tatsu: number } {
  let maxTotal = 0;
  let maxMelds = 0;
  let maxTatsu = 0;

  function backtrack(index: number, melds: number, tatsu: number) {
    // スキップして次のインデックスへ
    while (index < 38 && counts[index] === 0) index++;

    if (index >= 38) {
      if (melds * 2 + tatsu > maxTotal) {
        maxTotal = melds * 2 + tatsu;
        maxMelds = melds;
        maxTatsu = tatsu;
      }
      return;
    }

    // 刻子
    if (counts[index] >= 3) {
      counts[index] -= 3;
      backtrack(index, melds + 1, tatsu);
      counts[index] += 3;
    }

    // 順子
    if (index < 27 && index % 10 < 7 && counts[index] > 0 && counts[index + 1] > 0 && counts[index + 2] > 0) {
      counts[index]--;
      counts[index + 1]--;
      counts[index + 2]--;
      backtrack(index, melds + 1, tatsu);
      counts[index]++;
      counts[index + 1]++;
      counts[index + 2]++;
    }

    // 対子 (雀頭以外のパーツとしての対子)
    if (counts[index] >= 2) {
      counts[index] -= 2;
      backtrack(index, melds, tatsu + 1);
      counts[index] += 2;
    }

    // 塔子 (ペンチャン・リャンメン)
    if (index < 27 && index % 10 < 8 && counts[index] > 0 && counts[index + 1] > 0) {
      counts[index]--;
      counts[index + 1]--;
      backtrack(index, melds, tatsu + 1);
      counts[index]++;
      counts[index + 1]++;
    }

    // 塔子 (カンチャン)
    if (index < 27 && index % 10 < 7 && counts[index] > 0 && counts[index + 2] > 0) {
      counts[index]--;
      counts[index + 2]--;
      backtrack(index, melds, tatsu + 1);
      counts[index]++;
      counts[index + 2]++;
    }

    // 何も取らずに次へ
    const tmp = counts[index];
    counts[index] = 0;
    backtrack(index + 1, melds, tatsu);
    counts[index] = tmp;
  }

  backtrack(0, 0, 0);

  // 面子と塔子の合計は4つまで
  if (maxMelds + maxTatsu > 4) {
    maxTatsu = 4 - maxMelds;
  }

  return { melds: maxMelds, tatsu: maxTatsu };
}

function calculateStandardShanten(counts: number[]): number {
  let minShanten = 8;

  // 雀頭ありのケース
  for (let i = 0; i < 38; i++) {
    if (counts[i] >= 2) {
      counts[i] -= 2;
      const { melds, tatsu } = countMeldsAndTatsu([...counts]);
      const shanten = 8 - 2 * melds - tatsu - 1;
      minShanten = Math.min(minShanten, shanten);
      counts[i] += 2;
    }
  }

  // 雀頭なしのケース
  const { melds, tatsu } = countMeldsAndTatsu([...counts]);
  const shantenNoHead = 8 - 2 * melds - tatsu;
  minShanten = Math.min(minShanten, shantenNoHead);

  // 13枚の時は、アガリ状態(-1)を聴牌(0)に、聴牌(0)を1向聴(1)にスライドさせるためのオフセットは不要。
  // calculateShantenの中で、14枚なら和了(-1)、13枚なら最小でも聴牌(0)になるように計算される。
  // 現在のロジックでは 8 - 2*4 - 0 - 1 = -1 (和了) となる。
  // 13枚の時、8 - 2*melds - tatsu が最小 0 になるべき。
  // 例: 面子4つなら 8 - 2*4 = 0 (聴牌)。
  
  return minShanten;
}

function calculateChiitoiShanten(counts: number[]): number {
  let pairs = 0;
  let types = 0;
  const totalTiles = counts.reduce((a, b) => a + b, 0);
  for (let i = 0; i < 38; i++) {
    if (counts[i] >= 2) pairs++;
    if (counts[i] > 0) types++;
  }
  let shanten = 6 - pairs + Math.max(0, 7 - types);
  if (totalTiles === 14 && pairs === 7) return -1; // 14枚で7対子なら和了
  if (totalTiles === 13) shanten += 1;
  return shanten;
}

function calculateKokushiShanten(counts: number[]): number {
  const terminals = [0, 8, 10, 18, 20, 28, 30, 31, 32, 33, 34, 35, 36];
  let types = 0;
  let hasPair = false;
  const totalTiles = counts.reduce((a, b) => a + b, 0);

  for (const idx of terminals) {
    if (counts[idx] > 0) types++;
    if (counts[idx] >= 2) hasPair = true;
  }

  let shanten = 13 - types - (hasPair ? 1 : 0);
  if (totalTiles === 14 && types === 13 && hasPair) return -1; // 14枚で国士無双なら和了
  if (totalTiles === 13) shanten += 0; // 13枚の時はそのまま
  return shanten;
}

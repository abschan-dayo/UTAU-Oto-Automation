"""Filename parsing. Original spelling and katakana aliases are never rewritten."""
from dataclasses import dataclass, field
from pathlib import Path
import re
import unicodedata


def hiragana(text):
    return ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in text)


VOWELS = {}
for vowel, row in zip('aiueo', ('あかがさざただなはばぱまやらわぁゃゎ',
                              'いきぎしじちぢにひびぴみりゐぃ',
                              'うくぐすずつづぬふぶぷむゆるゔぅゅ',
                              'えけげせぜてでねへべぺめれゑぇ',
                              'おこごそぞとどのほぼぽもよろをぉょ')):
    VOWELS.update(dict.fromkeys(row, vowel))
VOWELS['ん'] = 'n'
SMALL = 'ゃゅょぁぃぅぇぉゎ'


@dataclass
class Mora:
    text: str
    vowel: str | None
    kind: str


def classify(text):
    h = hiragana(text)
    c = h[0]
    if c in 'っー' or c not in VOWELS:
        return 'special'
    if c == 'ん':
        return 'moraic_nasal'
    if c in 'あいうえおゐゑを':
        return 'vowel'
    if c in 'なにぬねのまみむめも' or text[0] in 'ガギグゲゴ':
        return 'nasal'
    if c in 'やゆよわ':
        return 'glide'
    if c in 'らりるれろ':
        return 'tap'
    if c in 'ちつじぢづ':
        return 'affricate'
    if c in 'さしすせそざずぜぞはひふへほゔ':
        return 'fricative'
    return 'stop'


@dataclass
class ParsedName:
    filename: str
    reading: str
    moras: list[Mora]
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)


def parse_name(filename, override=None):
    # NFC composes dakuten without folding katakana or changing the real filename.
    stem = unicodedata.normalize('NFC', override if override is not None else Path(filename).stem)
    runs = re.findall(r'[ぁ-ゖァ-ヶー]+', stem)
    if not runs:
        raise ValueError('ファイル名を確認してください: かな録音列がありません。--name-map で指定できます')
    reading = max(runs, key=len)
    reasons = []
    confidence = 1.0
    if len(runs) != 1:
        raise ValueError('ファイル名を確認してください: かな列が複数あり、一意に抽出できません。--name-map で指定できます')
    moras = []
    for c in reading:
        h = hiragana(c)
        if h in SMALL and moras and moras[-1].kind not in ('special', 'moraic_nasal'):
            moras[-1].text += c
            moras[-1].vowel = VOWELS[h]
        else:
            kind = classify(c)
            if h in SMALL:
                kind = 'special'
            moras.append(Mora(c, VOWELS.get(h) if kind != 'special' else None, kind))
    if any(m.kind == 'special' for m in moras):
        confidence = min(confidence, .3)
        reasons.append('促音・長音記号・未対応かなあり。該当箇所と直後のVCVは出力せず要確認')
    if re.search(r'[一-龯]', stem):
        confidence = min(confidence, .3)
        reasons.append('漢字を含むため、かな抽出結果を要確認（息・特殊音の可能性）')
    return ParsedName(filename, reading, moras, confidence, reasons)


def aliases(parsed):
    for i, m in enumerate(parsed.moras):
        if m.kind == 'special' or (i and parsed.moras[i-1].vowel is None):
            continue
        prev = parsed.moras[i-1].vowel if i else '-'
        yield i, f'{prev} {m.text}'

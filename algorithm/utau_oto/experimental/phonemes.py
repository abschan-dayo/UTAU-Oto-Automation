"""Conservative kana dictionary. Unknown aliases never get a guessed sequence."""
import re
import unicodedata
from ..names import hiragana, parse_name, aliases

TABLE={}
for consonant,row in [('', 'あいうえお'),('k','かきくけこ'),('g','がぎぐげご'),('s','さしすせそ'),('z','ざじずぜぞ'),('t','たちつてと'),('d','だぢづでど'),('n','なにぬねの'),('h','はひふへほ'),('b','ばびぶべぼ'),('p','ぱぴぷぺぽ'),('m','まみむめも'),('r','らりるれろ')]:
    for vowel,kana in zip('aiueo',row): TABLE[kana]=(consonant,vowel)
TABLE.update({'し':('sh','i'),'ち':('ch','i'),'つ':('ts','u'),'じ':('j','i'),'ぢ':('j','i'),'づ':('z','u'),'ふ':('f','u'),
    'や':('y','a'),'ゆ':('y','u'),'よ':('y','o'),'わ':('w','a'),'を':('','o'),'ん':('','n'),'ゔ':('v','u')})
for base,c in [('き','ky'),('ぎ','gy'),('し','sh'),('じ','j'),('ち','ch'),('ぢ','j'),('に','ny'),('ひ','hy'),('び','by'),('ぴ','py'),('み','my'),('り','ry')]:
    for small,v in [('ゃ','a'),('ゅ','u'),('ょ','o')]:TABLE[base+small]=(c,v)
for kana,c,v in [('てぃ','t','i'),('でぃ','d','i'),('とぅ','t','u'),('どぅ','d','u'),('でゅ','dy','u'),('てゅ','ty','u'),('しぇ','sh','e'),('じぇ','j','e'),('ちぇ','ch','e'),('ふぁ','f','a'),('ふぃ','f','i'),('ふぇ','f','e'),('ふぉ','f','o'),('うぃ','w','i'),('うぇ','w','e'),('うぉ','w','o')]: TABLE[kana]=(c,v)

def normalize_alias(alias):
    text=unicodedata.normalize('NFC',alias).strip()
    return re.sub(r'\s*[A-G][#b]?\d+$','',text).strip()

def parse_alias(alias):
    parts=normalize_alias(alias).split()
    if len(parts)!=2 or parts[0] not in ('-','a','i','u','e','o','n'): raise ValueError('Unsupported VCV alias: '+alias)
    kana=hiragana(parts[1])
    if kana not in TABLE: raise ValueError('Unknown kana: '+parts[1])
    consonant,vowel=TABLE[kana]
    states=['sil' if parts[0]=='-' else 'v:'+parts[0]]
    if consonant:states.append('c:'+consonant)
    states.append('v:'+vowel)
    return dict(previous=parts[0],consonant=consonant,vowel=vowel,kana=kana,states=states)

def target_index(filename,alias,occurrence=0):
    parsed=parse_name(filename)
    target=normalize_alias(alias)
    matches=[i for i,a in aliases(parsed) if normalize_alias(a)==target]
    if occurrence>=len(matches): raise ValueError('Alias not mapped to recording sequence')
    return matches[occurrence],len(parsed.moras)

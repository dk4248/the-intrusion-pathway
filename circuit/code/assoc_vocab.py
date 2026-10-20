"""Association vocabularies A_s and A_d, defined WITHOUT reference to any model output.

The rule: a "cloud" is associated information in an activation. The vocabulary that
defines it must be fixed independently of the head/MLP being tested, and kept disjoint from
  g        the exact gold token
  T_c      the literal tokens of the context
  C_r      generic words for the QUERY's answer category
  B_r      the other valid answers of the query's relation

Two kinds of association are measured separately, because they are different claims:
  A_s^attr(s)   ATTRIBUTES of the subject   - Argentina -> {Peso, Spanish}
                source: LRE's own cross-relation links over all 47 relations. Model-free.
  A_d^cat(d)    CO-CATEGORY of the distractor - brown -> {red, green, ..., colour, hue}
                source: the LRE range type the distractor came from (structural, not a
                judgement call) + a hand-written list of generic category words per type.
                d itself is always removed: that is the source-identity control.

CATEGORY_WORDS is the only hand-written part; each entry was checked against WordNet and
against the actual distractor inventory.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from lre_pool import load_all_relations, load_relations

# ---- hand-curated generic words per LRE range type ------------------------------------
# WordNet supplies co-hyponyms but its default senses are wrong for several of these
# ('hammer' -> gunlock, 'neutral' -> a person in a war, and the sentiment senses of
# positive/negative do not exist as nouns), so the generic terms are written out.
CATEGORY_WORDS = {
    "color":      ["color", "colour", "hue", "shade", "pigment", "colored", "coloured"],
    "sentiment":  ["sentiment", "tone", "emotion", "feeling", "attitude", "polarity"],
    "occupation": ["occupation", "profession", "job", "career", "worker", "professional"],
    "occupation type": ["occupation", "profession", "job", "career", "worker", "professional"],
    "tool":       ["tool", "implement", "instrument", "device", "equipment", "utensil"],
    "city":       ["city", "town", "capital", "metropolis", "municipality"],
    "country":    ["country", "nation", "state", "republic", "territory"],
    "language":   ["language", "tongue", "dialect", "speech", "spoken"],
    "currency":   ["currency", "money", "coin", "banknote", "cash"],
    "continent":  ["continent", "landmass", "region"],
    "superclass": ["category", "class", "kind", "type", "group"],
    "location":   ["location", "place", "site", "venue", "premises"],
}

_ALL = load_all_relations()
_OURS = load_relations()

# range type -> all surface forms of that type, pooled over the 47 LRE relations
_TYPE_MEMBERS = defaultdict(set)
for _r in _ALL.values():
    if _r["range"]:
        _TYPE_MEMBERS[_r["range"]].update(_r["objects"])
    if _r["domain"]:
        _TYPE_MEMBERS[_r["domain"]].update(_r["subjects"])

# subject -> its attributes across every LRE relation
_SUBJ_ATTR = defaultdict(set)
for _r in _ALL.values():
    for _s, _o in zip(_r["subjects"], _r["objects"]):
        _SUBJ_ATTR[_s.lower()].add(_o)


def _casings(w):
    """LRE is inconsistent about case ('Peso' vs our lowercased gold 'peso'), and the
    unembedding direction of ' Peso' differs from ' peso'. Carry both."""
    w = str(w).strip()
    if not w:
        return []
    out = {w}
    out.add(w[0].upper() + w[1:])
    out.add(w[0].lower() + w[1:])
    return sorted(out)


def expand_casings(words):
    out = []
    for w in words:
        out.extend(_casings(w))
    return sorted(set(out))


def norm_set(xs):
    return {x.strip() for x in xs if x and x.strip()}


def subject_attr_cloud(subject, exclude=()):
    """A_s^attr: what LRE says this entity is associated with, minus anything excluded."""
    ex = {e.lower() for e in exclude}
    return expand_casings(o for o in _SUBJ_ATTR.get(subject.lower(), set())
                          if o.lower() not in ex)


def category_cloud(word, type_name, exclude=(), max_members=60):
    """A_d^cat: co-members of the word's own category + generic category words.
    `word` itself is always excluded -- that is the source-identity control."""
    ex = {e.lower() for e in exclude} | {str(word).lower()}
    members = [m for m in sorted(_TYPE_MEMBERS.get(type_name, ()))
               if m.lower() not in ex]
    generic = [g for g in CATEGORY_WORDS.get(type_name, []) if g.lower() not in ex]
    return expand_casings(set(members[:max_members]) | set(generic))


def query_category_words(relation):
    """C_r: generic words for the QUERY's answer category (a control set, not A_s/A_d)."""
    return CATEGORY_WORDS.get(_OURS[relation]["range"], [])


def other_valid_answers(relation, gold, max_n=60):
    """B_r: the relation's other answers (a control set)."""
    g = str(gold).lower()
    return expand_casings(sorted({o for _, o in _OURS[relation]["facts"]
                                  if o.lower() != g})[:max_n])


# ---- Geva-scale attribute sets from Wikipedia summaries (see 09_fetch_wiki_attrs.py) ----
_WIKI_PATH = Path(__file__).resolve().parent.parent / "data" / "wiki_attrs.json"
_WIKI = json.load(_WIKI_PATH.open()) if _WIKI_PATH.exists() else {}


def wiki_attr_cloud(entity, exclude=(), max_n=150):
    """Content-word vocabulary of the entity's Wikipedia summary, most frequent first.

    This is the large, entity-specific, model-independent attribute set that Geva et al.'s
    attributes-rate metric needs; the LRE cross-link cloud (2-4 tokens) is far too small for
    a top-k membership rate. Disambiguation pages and misses return [].
    """
    import re
    from collections import Counter
    rec = _WIKI.get(entity)
    if not rec or not rec.get("extract") or rec.get("type") == "disambiguation":
        return []
    ex = {e.lower() for e in exclude} | {w.lower() for w in re.findall(r"[A-Za-z']+", str(entity))}
    from assoc_vocab import STOPWORDS_FOR_ATTRS as _STOP
    words = re.findall(r"[A-Za-z']+", rec["extract"])
    keep = [w for w in words if len(w) > 2 and w.lower() not in _STOP and w.lower() not in ex]
    return [w for w, _ in Counter(keep).most_common(max_n)]


STOPWORDS_FOR_ATTRS = set("""a an the of in on at to for from by with and or but is are was were
be been being this that these those it its as not no there here which who whom whose what when
where why how also than then them they he she his her him their our your my me we you i s t don
can will just into over under after before between during about against above below up down out
off again further once all any both each few more most other some such only own same so too very
one two three first second new used using use known called named including include includes may
might must shall should would could has have had do does did""".split())

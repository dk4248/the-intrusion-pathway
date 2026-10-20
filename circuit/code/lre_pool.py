"""Shared LRE pool: the selected relations, their templates, and fact loading.

Query template  : how the model is asked (answer must be the immediate continuation).
Context template: how a fact is stated as a prefix sentence  ->  "<ctx_tpl(subj)> <object>."

`family` groups relations that share a DOMAIN, so the same subject can appear in two
relations -> that is what makes the "near irrelevant" condition (same entity, different
relation, no answer to the query) constructible.
`range_name` is the type of the answer; range-disjointness is our structural leakage filter.
"""
from pathlib import Path
import json, re
REPO = Path(__file__).resolve().parents[2]  # repository root

def _unescape(s):
    """Some LRE entries are double-escaped in the source JSON, so json.load yields the
    literal 11-character string 'Bogot\\u00e1' instead of 'Bogotá'. Decode those."""
    if "\\u" in s:
        try:
            return s.encode("utf-8").decode("unicode_escape")
        except Exception:
            return s
    return s

LRE_ROOT = REPO / "external/lre_dataset"
# name -> (file, query_template_override|None, family, note)
# query_template_override is used where LRE's zero-shot template leaks the answer set.
RELATIONS = {
    # query_tpl / gold_lower chosen empirically by a template probe (mean rank-1 over 4 models)
    # query_ok=False -> fails as a query for these models, but still fine as a CONTEXT source
    #                   (it contributes range diversity: sentiment, occupation).
    "country_capital_city": dict(file="factual/country_capital_city.json", family="country",
        query_tpl="The capital of {} is the city of", probe=0.97),
    "country_language":     dict(file="factual/country_language.json", family="country",
        query_tpl="The main language spoken in {} is", probe=0.95),
    "city_in_country":      dict(file="factual/city_in_country.json", family="city",
        query_tpl="The city of {} is located in the country of", probe=0.78),
    "country_currency":     dict(file="factual/country_currency.json", family="country",
        query_tpl="In {}, people pay with a currency called the", gold_lower=True, probe=0.69),
    "task_done_by_tool":    dict(file="commonsense/task_done_by_tool.json", family="task",
        query_tpl="What tool is used for {}? Usually, you need a", probe=0.63),
    "landmark_in_country":  dict(file="factual/landmark_in_country.json", family="landmark",
        query_tpl="{} is in the country of", probe=0.61),
    "fruit_inside_color":   dict(file="commonsense/fruit_inside_color.json", family="fruit",
        query_tpl="What color are {} on the inside? They are", probe=0.54),
    "task_done_by_person":  dict(file="commonsense/task_done_by_person.json", family="task",
        query_tpl="The task of {} would be best performed by someone with the role of a", probe=0.51),
    "fruit_outside_color":  dict(file="commonsense/fruit_outside_color.json", family="fruit",
        query_tpl="The color of {} on the outside is", probe=0.24),
    # context-only from here down
    "word_sentiment":       dict(file="commonsense/word_sentiment.json", family="word",
        query_tpl="The sentiment of the word '{}' is", probe=0.18, query_ok=False),
    "person_occupation":    dict(file="factual/person_occupation.json", family="person",
        query_tpl="By profession, {} is a", probe=0.02, query_ok=False),
}


def load_relations(names=None):
    """-> {name: {subjects, objects, facts:[(subj,obj)], query_tpl, ctx_tpl, range, domain, family}}"""
    out = {}
    for name in (names or RELATIONS):
        cfg = RELATIONS[name]
        d = json.load((LRE_ROOT / cfg["file"]).open())
        props = d.get("properties", {})
        ctx_tpl = d["prompt_templates"][0]
        query_tpl = cfg.get("query_tpl") or (d.get("prompt_templates_zs") or d["prompt_templates"])[0]
        # dedupe by subject, keep first object
        facts, seen = [], set()
        for s in d["samples"]:
            if s["subject"] in seen:
                continue
            seen.add(s["subject"])
            facts.append((_unescape(s["subject"]), _unescape(s["object"])))
        if cfg.get("gold_lower"):
            facts = [(sub, o[0].lower() + o[1:] if o else o) for sub, o in facts]
        out[name] = dict(
            name=name, facts=facts, query_ok=cfg.get("query_ok", True), probe=cfg.get("probe"),
            objects=sorted({o for _, o in facts}),
            query_tpl=query_tpl, ctx_tpl=ctx_tpl,
            range=props.get("range_name", ""), domain=props.get("domain_name", ""),
            family=cfg["family"],
        )
    return out


def query_prompt(rel, subject):
    return rel["query_tpl"].format(subject)


_VOWEL_SOUND = re.compile(r"^(a|e|i|o|u)", re.I)

def _fix_article(stem, obj):
    """LRE templates hard-code 'a' ('works as a', 'is called a'), which yields
    'a actor' / 'a oven'. Fix agreement in CONTEXT sentences only -- query templates are
    left frozen, because choosing 'an' there would leak that the answer starts with a vowel."""
    if stem.endswith(" a") and _VOWEL_SOUND.match(obj.strip()):
        return stem + "n"
    return stem


def context_sentence(rel, subject, obj):
    """Fact -> one declarative prefix sentence. Handles a/an agreement and avoids a
    doubled period on objects that already end in one (e.g. 'Washington D.C.')."""
    stem = _fix_article(rel["ctx_tpl"].format(subject), obj)
    body = f"{stem} {obj}"
    if body and body[0].islower():        # subjects like 'researching history' start a sentence
        body = body[0].upper() + body[1:]
    return body if body.endswith(".") else body + "."


def load_all_relations():
    """Every LRE relation (47), used only to build a type lexicon for leakage checks."""
    import glob, os
    out = {}
    for f in glob.glob(str(LRE_ROOT / "*" / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        props = d.get("properties", {})
        name = os.path.basename(f)[:-5]
        out[name] = dict(
            name=name,
            domain=props.get("domain_name", ""), range=props.get("range_name", ""),
            subjects=[_unescape(s["subject"]) for s in d.get("samples", [])],
            objects=[_unescape(s["object"]) for s in d.get("samples", [])],
        )
    return out


def type_lexicon():
    """range/domain type -> set of surface strings of that type, pooled over all 47 relations.
    Used to ask: does this Wikipedia paragraph contain an entity of the query's ANSWER type?"""
    from collections import defaultdict
    lex = defaultdict(set)
    for rel in load_all_relations().values():
        if rel["range"]:
            lex[rel["range"]].update(o for o in rel["objects"] if len(o) >= 4)
        if rel["domain"]:
            lex[rel["domain"]].update(s for s in rel["subjects"] if len(s) >= 4)
    return {k: sorted(v) for k, v in lex.items()}

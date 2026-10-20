"""Build Geva-scale attribute sets from Wikipedia summaries.

Geva et al. (2023) measure an "attributes rate": the fraction of the top-k tokens of a
sublayer's vocabulary projection that belong to the subject's attribute set. That metric only
has power when the attribute set is large. Our LRE-derived clouds hold 2-4 tokens per subject,
so a top-20 rate is ~0 by construction whatever the mechanism.

Here the attribute set for an entity is the content-word vocabulary of its Wikipedia summary:
model-independent, entity-specific, and large enough for a rate metric. Excluded from every
set: stopwords, the entity's own surface tokens, and (at use time) the query's gold token.
Everything is cached so a run never depends on the network.
"""
import json, re, sys, time, urllib.request, urllib.parse
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "wiki_attrs.json"
UA = {"User-Agent": "mi-research-bot/1.0 (academic interpretability research)"}

STOP = set("""a an the of in on at to for from by with and or but is are was were be been being
this that these those it its as not no there here which who whom whose what when where why how
also than then them they he she his her him their our your my me we you i s t don can will just
into over under after before between during about against above below up down out off again
further once all any both each few more most other some such only own same so too very one two
three first second new used using use known called named including include includes may might
must shall should would could has have had do does did n't 're 've 'll 'd 'm""".split())

def summary(title, tries=5):
    """REST summary with backoff. The endpoint rate-limits after ~100 rapid requests and
    then returns 429 for everything, which looks exactly like a bad title if unhandled."""
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + urllib.parse.quote(str(title).replace(" ", "_")))
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.loads(urllib.request.urlopen(req, timeout=20).read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 503):
                time.sleep(2.0 * (t + 1))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(1.0 * (t + 1))
    raise last

def content_tokens(text, entity):
    ent = {w.lower() for w in re.findall(r"[A-Za-z']+", str(entity))}
    words = re.findall(r"[A-Za-z']+", text)
    keep = [w for w in words
            if len(w) > 2 and w.lower() not in STOP and w.lower() not in ent]
    return [w for w, _ in Counter(keep).most_common()]

def main():
    cache = json.load(CACHE.open()) if CACHE.exists() else {}
    S = json.load((ROOT / "data" / "pairs_v1.json").open())["samples"]
    wanted = {s["subject"] for s in S}
    for s in S:
        for v in s["variants"].get("far_irr", []):
            if v.get("distractor"):
                wanted.add(v["distractor"])
    todo = sorted(w for w in wanted
                  if w not in cache or not cache[w].get("extract"))   # retry failures
    print(f"{len(wanted)} entities, {len(todo)} to fetch", flush=True)
    for i, w in enumerate(todo):
        try:
            d = summary(w)
            cache[w] = dict(title=d.get("title"), type=d.get("type"),
                            extract=d.get("extract", ""),
                            desc=d.get("description", ""))
        except Exception as e:
            code = getattr(e, "code", "")
            cache[w] = dict(title=None, type="error", extract="",
                            desc=f"{type(e).__name__}{code}")
        time.sleep(0.4)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
            json.dump(cache, CACHE.open("w"), indent=1)
    json.dump(cache, CACHE.open("w"), indent=1)
    ok = sum(1 for v in cache.values() if v.get("extract"))
    disamb = sum(1 for v in cache.values() if v.get("type") == "disambiguation")
    print(f"cached {len(cache)}  with text: {ok}  disambiguation pages: {disamb}  -> {CACHE}")

if __name__ == "__main__":
    main()

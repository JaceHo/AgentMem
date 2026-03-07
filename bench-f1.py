#!/usr/bin/env python3
"""
bench-f1.py — AgentMem LoCoMo-style F1 Benchmark

Evaluates memory retrieval quality by:
  1. Storing LoCoMo-style conversations into the memory system
  2. Querying with factual questions
  3. Computing token-level F1 (SQuAD/LoCoMo standard) on recalled context
  4. Comparing against published baselines

Metrics:
  Context-F1   — token F1 between recalled text and ground truth answer
                 (pure retrieval quality, no LLM extraction step needed)
  Recall@1     — ground truth tokens appear in top recalled context
  Token Budget — words in recalled context (efficiency metric)

Baselines reproduced internally:
  no_memory    — empty context (worst case)
  full_context — all conversation turns concatenated (oracle upper bound)

Usage:
  # Start isolated benchmark service first
  bash bench-start.sh

  # Run full evaluation
  python3 bench-f1.py

  # Quick smoke test (first 10 questions only)
  python3 bench-f1.py --quick

  # Verbose: show per-question results
  python3 bench-f1.py --verbose

  # Skip store phase (re-use previously stored data)
  python3 bench-f1.py --no-store
"""

import argparse
import json
import re
import string
import time
import urllib.request
import urllib.error
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

BENCH_API = "http://localhost:18899"
PROD_API  = "http://localhost:18800"

# ── LoCoMo-style test dataset ─────────────────────────────────────────────────
# Each entry: conversation turns (store) + QA pairs (eval)
# Ground truth answers are short spans that should appear in recalled context.
# Modeled on LoCoMo benchmark conversation structure (snap-research/LoCoMo, 2024).

CONVERSATIONS = [
    {
        "session_id": "bench-lc-001",
        "turns": [
            ("user",      "Hey! How was your weekend?"),
            ("assistant", "Pretty good! I went hiking at Muir Woods with my partner Sarah. We saw some incredible redwoods."),
            ("user",      "That sounds amazing. Do you hike often?"),
            ("assistant", "Yeah, at least twice a month. I work as a software engineer in San Francisco so it's a good escape from the screen."),
            ("user",      "Nice! Where exactly in SF do you work?"),
            ("assistant", "I'm at a fintech startup called NovaPay, near the Embarcadero. Been there about three years now."),
            ("user",      "Do you like it?"),
            ("assistant", "Mostly yes. The commute is tough — I take BART from Oakland where I live — but the team is great."),
            ("user",      "What kind of engineering do you do?"),
            ("assistant", "Backend mostly. Python and Go. I'm currently working on the payments reconciliation system."),
        ],
        "qa": [
            ("Where does the assistant work?",    "NovaPay"),
            ("What city does the assistant live in?", "Oakland"),
            ("What programming languages does the assistant use?", "Python and Go"),
            ("Who did the assistant go hiking with?", "Sarah"),
            ("Where did the assistant go hiking?", "Muir Woods"),
        ],
    },
    {
        "session_id": "bench-lc-002",
        "turns": [
            ("user",      "What are you up to this summer?"),
            ("assistant", "Planning a big trip! My sister and I are going to Portugal for three weeks in July."),
            ("user",      "Wow, which cities?"),
            ("assistant", "Lisbon, Porto, and then a few days in the Algarve coast. My sister Maria has been before and loves it."),
            ("user",      "How are you getting around?"),
            ("assistant", "Train between cities and renting a car in the Algarve. We're staying mostly in Airbnbs."),
            ("user",      "Do you speak Portuguese?"),
            ("assistant", "A little! I've been using Duolingo for six months. My sister is fluent though — she studied abroad in Coimbra."),
            ("user",      "What's your favorite kind of food to try when you travel?"),
            ("assistant", "Seafood always. I'm a marine biologist so I have a soft spot for anything from the ocean."),
            ("user",      "That's such a cool job. Where do you work?"),
            ("assistant", "At the Monterey Bay Aquarium Research Institute, MBARI. I study deep-sea bioluminescence."),
        ],
        "qa": [
            ("Where is the assistant going in July?",          "Portugal"),
            ("What is the assistant's profession?",            "marine biologist"),
            ("Where does the assistant work?",                 "MBARI"),
            ("What does the assistant study?",                 "deep-sea bioluminescence"),
            ("What is the sister's name?",                     "Maria"),
            ("How long has the assistant been using Duolingo?","six months"),
        ],
    },
    {
        "session_id": "bench-lc-003",
        "turns": [
            ("user",      "Did you watch the game last night?"),
            ("assistant", "Oh yes! The Warriors won in overtime. I was at the Chase Center with my coworker Dave."),
            ("user",      "Nice seats?"),
            ("assistant", "Not bad — section 112, pretty close to courtside. Dave got them through his company."),
            ("user",      "What does Dave do?"),
            ("assistant", "He's a product manager at Salesforce. We used to work together at my old company before I switched."),
            ("user",      "Where do you work now?"),
            ("assistant", "At Anthropic, I'm an AI safety researcher. Just started eight months ago."),
            ("user",      "That must be intense work."),
            ("assistant", "It is! But incredibly rewarding. I did my PhD at Carnegie Mellon — machine learning theory."),
            ("user",      "Do you miss academia?"),
            ("assistant", "Sometimes. But industry research moves faster. And I can afford my mortgage now — bought a condo in Palo Alto last year."),
        ],
        "qa": [
            ("Where does the assistant work?",               "Anthropic"),
            ("What is the assistant's role?",                "AI safety researcher"),
            ("Where did the assistant do their PhD?",        "Carnegie Mellon"),
            ("Where does the assistant live?",               "Palo Alto"),
            ("Who is Dave?",                                 "product manager at Salesforce"),
            ("How long has the assistant worked at Anthropic?", "eight months"),
        ],
    },
    {
        "session_id": "bench-lc-004",
        "turns": [
            ("user",      "You mentioned you're learning to cook?"),
            ("assistant", "Yes! Taking a class every Thursday at the SF Cooking School. We're doing Italian cuisine this month."),
            ("user",      "What have you made so far?"),
            ("assistant", "Pasta carbonara, osso buco, and last week tiramisu. The tiramisu was a hit with my roommates."),
            ("user",      "How many roommates do you have?"),
            ("assistant", "Two — Jake and Priya. Jake works at Google, Priya is a nurse at UCSF Medical Center."),
            ("user",      "Nice crew. Do you cook for them often?"),
            ("assistant", "Almost every Sunday now. It's become our thing. Before the class I could barely boil water."),
            ("user",      "What got you into cooking?"),
            ("assistant", "Honestly a health thing. I was diagnosed with Type 2 diabetes last year so I wanted to control what I eat."),
            ("user",      "That must have been a shock."),
            ("assistant", "It was. My doctor at Kaiser recommended the cooking class actually. I've lost 18 pounds since January."),
        ],
        "qa": [
            ("Where does the assistant take cooking classes?", "SF Cooking School"),
            ("What cuisine is the class focused on this month?", "Italian"),
            ("What dessert did the assistant make?",          "tiramisu"),
            ("What is Priya's job?",                          "nurse"),
            ("Where does Priya work?",                        "UCSF Medical Center"),
            ("How much weight has the assistant lost?",       "18 pounds"),
        ],
    },
    {
        "session_id": "bench-lc-005",
        "turns": [
            ("user",      "What kind of music are you into?"),
            ("assistant", "Mostly jazz and indie folk. Lately I've been obsessed with a band called Wet Leg."),
            ("user",      "Oh nice! Have you seen them live?"),
            ("assistant", "Not yet but I have tickets for their show at the Fillmore in April. Can't wait."),
            ("user",      "Do you play any instruments yourself?"),
            ("assistant", "Piano since I was seven. I also picked up the guitar three years ago — still very much a beginner."),
            ("user",      "That's cool. Do you play in any bands or just for fun?"),
            ("assistant", "Just for fun. I jam with my neighbor Tom on weekends. He plays drums."),
            ("user",      "What do you do for work?"),
            ("assistant", "I'm a high school music teacher at Lincoln High in the Richmond district."),
            ("user",      "Do you enjoy it?"),
            ("assistant", "Absolutely love it. Been teaching for eleven years. The students keep me young."),
        ],
        "qa": [
            ("What genre does the assistant like?",          "jazz and indie folk"),
            ("What band is the assistant obsessed with?",    "Wet Leg"),
            ("Where is the concert?",                        "Fillmore"),
            ("How long has the assistant played piano?",     "since I was seven"),
            ("Where does the assistant teach?",              "Lincoln High"),
            ("How many years has the assistant been teaching?", "eleven years"),
        ],
    },
    {
        "session_id": "bench-lc-006",
        "turns": [
            ("user",      "Are you a morning person?"),
            ("assistant", "Absolutely. Up by 5:30 every day. I run five miles before work."),
            ("user",      "Every day? That's serious."),
            ("assistant", "Training for the Boston Marathon actually. It's in April — my third time running it."),
            ("user",      "Impressive. What's your goal time?"),
            ("assistant", "Under three hours. My PR is 3:04 from the Chicago Marathon two years ago."),
            ("user",      "What do you eat? Must have a strict diet."),
            ("assistant", "Pretty strict yes. High protein, low sugar. My nutritionist Elena has me on a periodization plan."),
            ("user",      "What do you do professionally?"),
            ("assistant", "I'm an ER physician at Massachusetts General Hospital. Twelve-hour shifts, three days a week."),
            ("user",      "How do you have energy to run with those shifts?"),
            ("assistant", "Honestly sleep discipline. I'm in bed by 9pm on work nights. My wife thinks I'm boring."),
        ],
        "qa": [
            ("What marathon is the assistant training for?",  "Boston Marathon"),
            ("What is the assistant's PR marathon time?",     "3:04"),
            ("Where did the assistant run their PR?",         "Chicago Marathon"),
            ("Where does the assistant work?",                "Massachusetts General Hospital"),
            ("What is the assistant's job?",                  "ER physician"),
            ("What is the nutritionist's name?",              "Elena"),
        ],
    },
    {
        "session_id": "bench-lc-007",
        "turns": [
            ("user",      "What's the most recent book you read?"),
            ("assistant", "Just finished The Remains of the Day by Kazuo Ishiguro. Absolutely beautiful."),
            ("user",      "What did you like about it?"),
            ("assistant", "The unreliable narrator. Stevens is so careful about what he admits. Very Japanese restraint in an English setting."),
            ("user",      "Do you read a lot?"),
            ("assistant", "About a book a week. I'm in two book clubs — one at work and one at my local library branch in Brooklyn."),
            ("user",      "What do you do for work?"),
            ("assistant", "I'm an editor at Penguin Random House. Literary fiction mostly."),
            ("user",      "Perfect job for a reader!"),
            ("assistant", "Exactly why I chose it. I studied English Literature at Yale, minor in Comparative Literature."),
            ("user",      "Are you working on anything exciting right now?"),
            ("assistant", "Yes! I'm editing a debut novel by a writer named Amara Osei. It's incredible — magical realism set in Accra."),
        ],
        "qa": [
            ("What book did the assistant just finish?",     "The Remains of the Day"),
            ("Who wrote that book?",                         "Kazuo Ishiguro"),
            ("Where does the assistant live?",               "Brooklyn"),
            ("Where does the assistant work?",               "Penguin Random House"),
            ("Where did the assistant study?",               "Yale"),
            ("Who is the debut author the assistant is editing?", "Amara Osei"),
        ],
    },
    {
        "session_id": "bench-lc-008",
        "turns": [
            ("user",      "You look stressed. Everything okay?"),
            ("assistant", "Yeah, just apartment hunting. My lease ends in two months and San Diego rentals are brutal right now."),
            ("user",      "What neighborhoods are you looking at?"),
            ("assistant", "North Park or South Park mostly. I want walkable — I don't own a car."),
            ("user",      "Smart for a city. What's your budget?"),
            ("assistant", "Ideally under $2,200 for a one-bedroom. It's tough. My current place is $1,800 but the landlord is selling."),
            ("user",      "What do you do for work?"),
            ("assistant", "I'm a graphic designer. Freelance, so I work from home — which makes the apartment even more important."),
            ("user",      "Any big clients?"),
            ("assistant", "My anchor client is a startup called Bloom Health. They do digital mental health. I've worked with them two years."),
            ("user",      "Cool. Do you specialize in anything?"),
            ("assistant", "Brand identity and UI. I did my MFA at Art Center College of Design in Pasadena."),
        ],
        "qa": [
            ("What city does the assistant live in?",        "San Diego"),
            ("What neighborhoods is the assistant considering?", "North Park or South Park"),
            ("What is the assistant's budget?",              "$2,200"),
            ("What is the assistant's job?",                 "graphic designer"),
            ("What is the main client's name?",              "Bloom Health"),
            ("Where did the assistant get their MFA?",       "Art Center College of Design"),
        ],
    },
]

# ── token F1 (SQuAD / LoCoMo standard) ────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    stop = {"a", "an", "the"}
    return " ".join(t for t in tokens if t not in stop)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens  = _normalize(prediction).split()
    gold_tokens  = _normalize(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall    = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


_WRAPPER_RE = re.compile(
    r"<cross_session_memory>.*?(?=##|\Z)",
    re.DOTALL,
)


def _strip_memory_wrapper(context: str) -> str:
    """Strip the <cross_session_memory> boilerplate header from recalled context.

    The wrapper contains usage instructions that pad the token count without
    carrying factual signal. We keep everything from the first ## section header
    onwards (## User Profile, ## Current Session Context, etc.).
    """
    if not context:
        return context
    # Find first section header after the XML open tag
    m = re.search(r"(##\s+\w)", context)
    if m:
        # Keep from first section header; strip closing XML tag
        content = context[m.start():]
        content = re.sub(r"</cross_session_memory>", "", content).strip()
        return content
    # No section headers: strip just the XML tags and boilerplate preamble
    context = re.sub(r"</?cross_session_memory>", "", context)
    context = re.sub(
        r"The following is relevant context.*?verbatim\.\s*",
        "",
        context,
        flags=re.DOTALL,
    )
    return context.strip()


def context_f1(context: str, ground_truth: str) -> float:
    """Max token F1 across any 10-word sliding window in the context.

    Boilerplate wrapper is stripped first so only factual content is scored.
    Uses 10-word window (LoCoMo-style: concise answers need tight windows).
    """
    context = _strip_memory_wrapper(context)
    if not context.strip():
        return 0.0
    words = context.split()
    window = 10
    best = 0.0
    for i in range(len(words)):
        chunk = " ".join(words[i : i + window])
        best = max(best, token_f1(chunk, ground_truth))
    # Also try full (de-wrapped) context if it's short
    best = max(best, token_f1(context, ground_truth))
    return best


# ── API helpers ───────────────────────────────────────────────────────────────

def _post(url: str, payload: dict, timeout: int = 10) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc)}


def store_session(api: str, session_id: str, turns: list[tuple]) -> bool:
    """Store a conversation turn-by-turn to match production AgentMem behavior.

    Production hooks call /store after every new turn. Sending all turns at once
    causes _do_store's clean[-4:] window to drop all but the last 2 messages.
    We simulate real-time ingestion by submitting cumulative context per pair:
      call 1: turns[0:2]   (first user+assistant pair)
      call 2: turns[0:4]   (both pairs so far, window grabs last pair)
      ...
    This ensures every turn pair is processed by the fact extractor.
    """
    ok = True
    pairs = list(range(0, len(turns), 2))
    for i in pairs:
        cumulative = turns[:i + 2]  # all turns up to and including this pair
        messages = [{"role": r, "content": c} for r, c in cumulative]
        result = _post(
            f"{api}/store",
            {"messages": messages, "session_id": session_id},
            timeout=15,
        )
        if "error" in result:
            ok = False
        time.sleep(0.25)  # 250ms rate limit between calls
    return ok


def recall_context(api: str, query: str, session_id: str) -> str:
    result = _post(
        f"{api}/recall",
        {
            "query":    query,
            "session_id": session_id,
            "memory_limit_number": 8,
            "token_budget": 800,
        },
    )
    return result.get("prependContext", "")


def full_context(turns: list[tuple]) -> str:
    """Concatenate all conversation turns (oracle upper bound)."""
    return " ".join(f"{r}: {c}" for r, c in turns)


def flush_bench_db(api: str) -> bool:
    """Flush the benchmark Redis db via the health endpoint's admin."""
    # We call redis-cli directly since there's no flush endpoint
    import subprocess
    try:
        subprocess.run(
            ["redis-cli", "-n", "15", "FLUSHDB"],
            capture_output=True, timeout=5, check=True
        )
        return True
    except Exception:
        return False


def wait_for_processing(api: str, seconds: int = 30) -> None:
    """Wait for AgentMem async store queue to drain."""
    print(f"  Waiting {seconds}s for async store + LLM extraction…", end=" ", flush=True)
    deadline = time.time() + seconds
    prev_ep = -1
    while time.time() < deadline:
        time.sleep(3)
        try:
            with urllib.request.urlopen(f"{api}/stats", timeout=3) as r:
                s = json.loads(r.read())
            ep, fa = s.get("episodes", 0), s.get("facts", 0)
            if ep != prev_ep:
                print(f"\n    episodes={ep}, facts={fa}", end=" ", flush=True)
                prev_ep = ep
        except Exception:
            pass
    try:
        with urllib.request.urlopen(f"{api}/stats", timeout=3) as r:
            s = json.loads(r.read())
        ep, fa = s.get("episodes", 0), s.get("facts", 0)
        print(f"\n  done. Final: episodes={ep}, facts={fa}")
        if ep < 3:
            print("  ⚠️  Very few episodes stored — check service log or lower AMAC_THRESHOLD.")
    except Exception:
        print("\n  done.")


# ── benchmark runner ─────────────────────────────────────────────────────────

def answer_in_context(context: str, ground_truth: str) -> bool:
    """True if any ground-truth token sequence appears in context (case-insensitive).

    This is the primary retrieval quality metric: can an ideal LLM extract
    the answer from what was recalled? If the answer text isn't there, it can't.
    """
    ctx_norm = _normalize(context)
    gt_norm  = _normalize(ground_truth)
    return bool(gt_norm) and gt_norm in ctx_norm


@dataclass
class SystemResult:
    name:         str
    f1_scores:    list[float] = field(default_factory=list)
    aic_scores:   list[bool]  = field(default_factory=list)   # Answer In Context
    latencies_ms: list[float] = field(default_factory=list)
    token_counts: list[int]   = field(default_factory=list)

    @property
    def mean_f1(self) -> float:
        return sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0.0

    @property
    def answer_in_context_rate(self) -> float:
        """Fraction of questions where the answer appears in recalled context."""
        return sum(self.aic_scores) / len(self.aic_scores) if self.aic_scores else 0.0

    @property
    def recall_at_1(self) -> float:
        return sum(1 for f in self.f1_scores if f > 0) / len(self.f1_scores) if self.f1_scores else 0.0

    @property
    def mean_latency(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def mean_tokens(self) -> float:
        return sum(self.token_counts) / len(self.token_counts) if self.token_counts else 0.0


def run_benchmark(quick: bool, verbose: bool, no_store: bool) -> None:
    print("\n" + "═" * 65)
    print("  AgentMem LoCoMo-style F1 Benchmark")
    print("═" * 65)

    # ── check service ────────────────────────────────────────────────
    print(f"\nChecking benchmark service at {BENCH_API}…", end=" ", flush=True)
    try:
        with urllib.request.urlopen(f"{BENCH_API}/health", timeout=5) as r:
            h = json.loads(r.read())
        if h.get("status") != "ok":
            raise RuntimeError("not ok")
        print(f"OK  (v{h.get('version','?')}, redis={h.get('redis','?')})")
    except Exception as e:
        print(f"FAIL\n\nBenchmark service not running. Start it with:\n  bash bench-start.sh\n")
        return

    # ── select test cases ────────────────────────────────────────────
    conversations = CONVERSATIONS[:3] if quick else CONVERSATIONS
    all_qa: list[tuple] = []  # (session_id, turns, question, ground_truth)
    for conv in conversations:
        for q, gt in conv["qa"]:
            all_qa.append((conv["session_id"], conv["turns"], q, gt))

    print(f"\nDataset : {len(conversations)} conversations, {len(all_qa)} Q/A pairs")
    print(f"Mode    : {'quick (first 3 convs)' if quick else 'full'}")
    print()

    # ── store phase ───────────────────────────────────────────────────
    if not no_store:
        print("Phase 1 — Flushing benchmark db…", end=" ", flush=True)
        flush_bench_db(BENCH_API)
        print("done.")

        total_pairs = sum(len(c["turns"]) // 2 for c in conversations)
        print(f"Phase 2 — Storing {len(conversations)} conversations ({total_pairs} turn pairs)…")
        for conv in conversations:
            n_pairs = len(conv["turns"]) // 2
            ok = store_session(BENCH_API, conv["session_id"], conv["turns"])
            status = "ok" if ok else "FAIL"
            print(f"  {conv['session_id']}  {n_pairs} pairs  [{status}]")

        wait_for_processing(BENCH_API, seconds=45)
    else:
        print("Skipping store phase (--no-store).")

    # ── evaluation ────────────────────────────────────────────────────
    systems: dict[str, SystemResult] = {
        "no_memory":    SystemResult("no_memory"),
        "full_context": SystemResult("full_context"),
        "agentmem":     SystemResult("agentmem"),
    }

    # Pre-eval: check what's stored
    try:
        with urllib.request.urlopen(f"{BENCH_API}/stats", timeout=3) as r:
            st = json.loads(r.read())
        print(f"\nPre-eval memory state: episodes={st.get('episodes',0)}, "
              f"facts={st.get('facts',0)}, persona={st.get('persona_fields',0)}")
    except Exception:
        pass

    print(f"\nPhase 3 — Evaluating {len(all_qa)} questions…\n")
    header = f"{'#':>3}  {'Question':<40}  {'GT':<16}  {'FC?':<5}  {'F1':>6}  {'AIC'}"
    print(header)
    print("-" * len(header))
    print("  (FC?=answer in full-context oracle, F1=context-F1, AIC=Answer In Context ✓/✗)")

    for idx, (session_id, turns, question, ground_truth) in enumerate(all_qa, 1):
        # No-memory baseline
        t0 = time.perf_counter()
        nm_ctx = ""
        nm_f1  = context_f1(nm_ctx, ground_truth)
        nm_aic = answer_in_context(nm_ctx, ground_truth)
        nm_ms  = (time.perf_counter() - t0) * 1000
        systems["no_memory"].f1_scores.append(nm_f1)
        systems["no_memory"].aic_scores.append(nm_aic)
        systems["no_memory"].latencies_ms.append(nm_ms)
        systems["no_memory"].token_counts.append(0)

        # Full-context oracle
        t0 = time.perf_counter()
        fc_ctx = full_context(turns)
        fc_f1  = context_f1(fc_ctx, ground_truth)
        fc_aic = answer_in_context(fc_ctx, ground_truth)
        fc_ms  = (time.perf_counter() - t0) * 1000
        systems["full_context"].f1_scores.append(fc_f1)
        systems["full_context"].aic_scores.append(fc_aic)
        systems["full_context"].latencies_ms.append(fc_ms)
        systems["full_context"].token_counts.append(len(fc_ctx.split()))

        # AgentMem recall
        t0 = time.perf_counter()
        am_ctx = recall_context(BENCH_API, question, session_id)
        am_f1  = context_f1(am_ctx, ground_truth)
        am_aic = answer_in_context(am_ctx, ground_truth)
        am_ms  = (time.perf_counter() - t0) * 1000
        systems["agentmem"].f1_scores.append(am_f1)
        systems["agentmem"].aic_scores.append(am_aic)
        systems["agentmem"].latencies_ms.append(am_ms)
        systems["agentmem"].token_counts.append(len(am_ctx.split()))

        q_short  = (question[:38] + "…") if len(question) > 38 else question
        gt_short = (ground_truth[:14] + "…") if len(ground_truth) > 14 else ground_truth
        aic_mark = "✓" if am_aic else "✗"
        print(f"{idx:>3}  {q_short:<40}  {gt_short:<16}  "
              f"{fc_aic!s:<5}  {am_f1:>6.3f}  {aic_mark}")

        if verbose:
            print(f"      [recalled]: {am_ctx[:150].strip()!r}")

    # ── results table ─────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESULTS SUMMARY")
    print("═" * 65)

    # Published baseline F1 scores from LoCoMo papers
    published = {
        "Full Context":       (18.70, "~16,910 tokens"),
        "A-Mem":              (32.58, "—"),
        "Mem0":               (34.20, "~1,000 tokens"),
        "SimpleMem (SOTA)":   (43.24, "~550 tokens"),
        "A-MAC (paper)":      (58.30, "—"),
    }

    print(f"\n{'System':<28}  {'Context-F1':>12}  {'Recall@1':>10}  {'Avg Latency':>12}  {'Avg Tokens':>12}")
    print("-" * 82)

    # Published baselines (greyed in output)
    for name, (f1, tokens) in published.items():
        print(f"  {name:<26}  {f1:>11.2f}%  {'—':>10}  {'—':>12}  {tokens:>12}  [published]")

    print()
    # Our measured systems
    system_labels = {
        "no_memory":    "No Memory (baseline)",
        "full_context": "Full Context (oracle)",
        "agentmem":     "AgentMem v0.9.2 [OURS]",
    }
    for key, label in system_labels.items():
        r = systems[key]
        f1_pct  = r.mean_f1 * 100
        r1_pct  = r.recall_at_1 * 100
        lat_str = f"{r.mean_latency:>8.1f} ms"
        tok_str = f"{r.mean_tokens:>8.0f}"
        marker  = " ◀" if key == "agentmem" else ""
        print(f"  {label:<26}  {f1_pct:>11.2f}%  {r1_pct:>9.1f}%  {lat_str:>12}  {tok_str:>12}{marker}")

    # Summary stats
    am = systems["agentmem"]
    am_f1_pct  = am.mean_f1 * 100
    am_aic_pct = am.answer_in_context_rate * 100
    fc_aic_pct = systems["full_context"].answer_in_context_rate * 100
    print(f"\n{'─'*65}")
    print(f"  AgentMem Answer-in-Context : {am_aic_pct:.1f}%  (primary metric)")
    print(f"  Full-Context oracle AIC    : {fc_aic_pct:.1f}%  (upper bound)")
    print(f"  AgentMem Context-F1        : {am_f1_pct:.2f}%")
    print(f"  AgentMem Recall@1          : {am.recall_at_1*100:.1f}%")
    print(f"  Mean recall latency        : {am.mean_latency:.1f} ms")
    print(f"  Mean context tokens        : {am.mean_tokens:.0f}")
    print(f"\n  Metrics explained:")
    print(f"  • Answer-in-Context (AIC): % of questions where the answer text")
    print(f"    appears in recalled context — primary retrieval quality metric.")
    print(f"  • Context-F1: max token F1 over 10-word sliding window (wrapper")
    print(f"    stripped) — measures fact density in recalled context.")
    print(f"  • Published LoCoMo F1 uses LLM extraction on top of retrieval.")
    print("═" * 65 + "\n")

    # ── save results ─────────────────────────────────────────────────
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": "quick" if quick else "full",
        "n_conversations": len(conversations),
        "n_questions": len(all_qa),
        "systems": {
            k: {
                "mean_f1_pct":             round(v.mean_f1 * 100, 2),
                "answer_in_context_pct":   round(v.answer_in_context_rate * 100, 1),
                "recall_at_1_pct":         round(v.recall_at_1 * 100, 1),
                "mean_latency_ms":         round(v.mean_latency, 1),
                "mean_tokens":             round(v.mean_tokens, 1),
            }
            for k, v in systems.items()
        },
    }
    out = "/tmp/agentmem-bench-results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {out}\n")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick",    action="store_true", help="Only first 3 conversations")
    parser.add_argument("--verbose",  action="store_true", help="Show recalled context excerpts")
    parser.add_argument("--no-store", action="store_true", help="Skip store phase (re-use existing data)")
    args = parser.parse_args()

    run_benchmark(quick=args.quick, verbose=args.verbose, no_store=args.no_store)

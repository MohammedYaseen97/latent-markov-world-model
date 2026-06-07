"""check_assumptions.py — Incremental assumption testing for the latent Markov arm.

Run locally on RTX 4060 (8 GB VRAM).  Each step can be run independently.
Steps are added incrementally as the investigation deepens.

Usage:
    python scripts/check_assumptions.py --step traces [options]

Steps implemented:
    traces   — A0/A1/A9: generate atomic-step teacher traces and inspect them.
               Each chunk = one complete reasoning step ended by a natural EOS.
               Stops when \\boxed{} appears or max_chunks is reached.
               What we're checking:
                 A0  — does the teacher produce multiple distinct reasoning steps?
                 A1  — do chunk boundaries fall at natural semantic step ends?
                 A9  — does the teacher solve the problem correctly overall?
                 (implicit) — will the model follow the "one step, then stop" prompt?

Options:
    --n              number of problems            (default: 20)
    --seed           random seed                   (default: 42)
    --pool           path to JSONL pool            (default: data/math_easy_pool.jsonl)
    --levels         difficulty levels to sample   (default: 1 2 3 4)
    --model          HF repo id                    (default: Qwen/Qwen2.5-1.5B-Instruct)
    --revision       pinned commit                 (default: see base_model.yaml)
    --max-chunks     max reasoning steps per prob  (default: 12)
    --max-step-tok   max tokens per step           (default: 512)
    --min-step-tok   min tokens per step           (default: 10)
    --out            path to write JSON            (optional)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── repo root on sys.path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.grpo_baseline import answers_equivalent, extract_answer

# ── defaults ──────────────────────────────────────────────────────────────────
# Qwen2.5-1.5B — local RTX 4060 (fits in bfloat16, 8 GB VRAM)
DEFAULT_MODEL    = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

# Qwen3-8B — cloud / GPU with ≥20 GB VRAM (bfloat16), or local in 4-bit quant.
# Qwen3 supports a /think and /no_think suffix in the system prompt to toggle
# chain-of-thought.  We use /no_think so output is direct reasoning text, not
# wrapped in <think>…</think> blocks that would confuse chunk parsing.
# Unpin the revision once you know which commit you want.
# DEFAULT_MODEL    = "Qwen/Qwen3-8B"
# DEFAULT_REVISION = None  # or pin to a specific commit hash

DEFAULT_POOL     = REPO_ROOT / "data" / "math_easy_pool.jsonl"

# ── atomic-step system prompt ──────────────────────────────────────────────────
# Rule-list style: explicit STOP, box-on-same-step, no multi-step packing.
# Granularity hint gives the model a concrete example of what "one step" means.
# /no_think disables Qwen3's internal <think> wrapper (harmless on Qwen2.5).
_ATOMIC_SYSTEM_PROMPT = """\
Solve this problem one step at a time. /no_think

A step is: derive one new intermediate result and state it explicitly before stopping.
  - One step = one new fact established (one value computed, one rule applied, one equation reduced).
  - The step ends with that result written out as a complete sentence or expression — not mid-derivation.
  - Example of ONE step: "Applying the quadratic formula to x²+5x+6=0 gives roots x=-2 and x=-3."
  - NOT one step: solving the whole problem in a single response.
  - NOT one step: stopping mid-calculation before the intermediate result is stated.

Rules:
- Each response: perform exactly one step as defined above, state the result, then STOP.
- The moment the stated result IS the final answer, write \\boxed{answer} at the end of
  THAT SAME response and STOP. Do not save the box for a separate response.
- Never write "Step 1:", "Step 2:" headers — just write the step content directly.\
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_pool(paths: list[Path], levels: list[int] | None, n: int, seed: int) -> list[dict]:
    """Load problems from one or more JSONL pool files.

    levels=None means accept all difficulty values found in the pool.
    Multiple paths are merged before sampling.
    """
    rng = random.Random(seed)
    all_problems: list[dict] = []
    n_skipped_asy = 0
    for path in paths:
        with open(path) as f:
            for line in f:
                prob = json.loads(line)
                if levels is not None and prob.get("difficulty") not in levels:
                    continue
                # Skip problems that embed Asymptote diagrams — LLMs cannot
                # reliably parse layout code to reconstruct the figure.
                if "[asy]" in prob.get("prompt", ""):
                    n_skipped_asy += 1
                    continue
                all_problems.append(prob)
    if n_skipped_asy:
        print(f"[info] skipped {n_skipped_asy} problem(s) with [asy] diagrams", flush=True)
    if not all_problems:
        raise ValueError(
            f"No problems found in {[str(p) for p in paths]} "
            f"at levels {levels}. Check --pool and --levels."
        )
    if len(all_problems) < n:
        print(f"[warn] pool only has {len(all_problems)} problems "
              f"(levels={levels}); using all {len(all_problems)}")
        return all_problems
    return rng.sample(all_problems, n)


def _load_model(model_id: str, revision: str | None, device: torch.device
                ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    rev_label = revision[:8] if revision else "latest"
    print(f"Loading {model_id} @ {rev_label}… (bfloat16, device_map=auto)", flush=True)
    kwargs: dict = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    if revision is not None:
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    tok_kwargs = {"revision": revision} if revision is not None else {}
    tok = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  loaded — {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params",
          flush=True)
    return model, tok


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width,
                                       initial_indent=indent, subsequent_indent=indent))
    return "\n".join(lines)


def _boundary_quality(text: str) -> str:
    """Heuristic: does the chunk end at a natural stopping point?"""
    stripped = text.rstrip()
    if not stripped:
        return "EMPTY"
    last = stripped[-1]
    if last in ".!?":    return "GOOD (sentence end)"
    if last in "}])":    return "OK   (bracket/brace)"
    if last in ",:;":    return "POOR (mid-list)"
    if "\\boxed{" in stripped: return "FINAL (has \\boxed)"
    return f"UNKNOWN ('{last}')"


def _has_boxed(text: str) -> bool:
    return "\\boxed{" in text or r"\boxed{" in text


# ══════════════════════════════════════════════════════════════════════════════
#  Atomic-step teacher generation
#  One reasoning step per call, terminated by natural EOS.
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _generate_atomic_chunks(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: dict,
    max_chunks: int,
    max_step_tokens: int,
    min_step_tokens: int,
    device: torch.device,
) -> list[dict]:
    """Generate reasoning chunks for one problem using atomic-step prompting.

    The model is asked to produce exactly one reasoning step per turn, then stop.
    Context grows turn-by-turn (teacher has full history as in the original design).
    Stops when:
      - a chunk contains \\boxed{} (answer found — natural termination), or
      - max_chunks is reached (model failed to converge).

    Returns:
        list of dicts, one per chunk:
          {
            "text":    decoded text of the step,
            "n_tokens": number of generated tokens,
            "stopped_by": "boxed" | "eos" | "max_tokens" | "max_chunks",
          }
    """
    pad_id = tokenizer.eos_token_id

    # Build the initial prompt: system + user (problem statement)
    messages: list[dict] = [
        {"role": "system",    "content": _ATOMIC_SYSTEM_PROMPT},
        {"role": "user",      "content": problem["prompt"]},
    ]

    chunks: list[dict] = []

    for chunk_idx in range(max_chunks):
        # Encode the current conversation context.
        # enable_thinking=False suppresses Qwen3's <think> blocks so output is
        # plain reasoning text.  Qwen2.5 doesn't support this kwarg, so we fall
        # back gracefully.
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # Generate one step — let EOS stop it naturally
        output = model.generate(
            input_ids,
            max_new_tokens=max_step_tokens,
            min_new_tokens=min_step_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Extract only the newly generated tokens
        new_ids    = output[0, input_ids.shape[1]:]
        n_tokens   = new_ids.shape[0]
        step_text  = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Determine why generation stopped
        last_token = new_ids[-1].item()
        if last_token == tokenizer.eos_token_id:
            stopped_by = "eos"
        elif n_tokens >= max_step_tokens:
            stopped_by = "max_tokens"
        else:
            stopped_by = "eos"  # generate() can stop early for other reasons

        if _has_boxed(step_text):
            stopped_by = "boxed"

        chunks.append({
            "text":       step_text,
            "n_tokens":   n_tokens,
            "stopped_by": stopped_by,
        })

        # Extend the conversation context with the model's response
        messages.append({"role": "assistant", "content": step_text})

        # If the model produced the final answer, we're done
        if stopped_by == "boxed":
            break

        # Continuation message — explicitly gives the model permission to terminate.
        # "Next intermediate result." was wrong here: it's an imperative that forces
        # the model to produce something new, contradicting the system-prompt rule
        # "the moment the result IS the final answer, write \boxed{} and stop."
        #
        # After many turns the urgency escalates:
        #   turns 1..max_chunks-3  →  standard continuation (allows stop)
        #   last 2 turns           →  urgent: box if you have the answer
        #
        # Conversation shape at step N:
        #   [system] Solve one step at a time… rules…
        #   [user]   <problem>
        #   [asst]   <step 1>
        #   [user]   "Next step — or \boxed{answer} if done."
        #   [asst]   <step 2>
        #   [user]   "Next step — or \boxed{answer} if done."
        #   [asst]   <step N>    ← now generating
        steps_remaining = max_chunks - (chunk_idx + 1)
        if steps_remaining <= 2:
            continuation = (
                "If you have the final answer, write \\boxed{answer} now and stop. "
                "Otherwise, one more step only."
            )
        else:
            continuation = "Next step — or \\boxed{answer} if done."
        messages.append({"role": "user", "content": continuation})

    else:
        # max_chunks exhausted without \\boxed{}
        if chunks:
            chunks[-1]["stopped_by"] = "max_chunks"

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  Step: traces
# ══════════════════════════════════════════════════════════════════════════════

def run_traces(args: argparse.Namespace) -> None:
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problems = _load_pool(args.pool, args.levels, args.n, args.seed)
    model, tok = _load_model(args.model, args.revision, device)
    print(f"  Pool: {[str(p) for p in args.pool]}", flush=True)
    diff_counts: dict[int, int] = {}
    for prob in problems:
        d = prob.get("difficulty", "?")
        diff_counts[d] = diff_counts.get(d, 0) + 1
    print(f"  Sampled difficulties: {dict(sorted(diff_counts.items()))}", flush=True)

    print(f"\nGenerating atomic-step teacher traces for {len(problems)} problems", flush=True)
    print(f"  max_chunks={args.max_chunks}  "
          f"max_step_tokens={args.max_step_tokens}  "
          f"min_step_tokens={args.min_step_tokens}", flush=True)
    print(f"  System prompt:\n{_wrap(_ATOMIC_SYSTEM_PROMPT, width=90)}\n", flush=True)

    all_records: list[dict] = []

    for idx, prob in enumerate(problems):
        print(f"{'─'*80}", flush=True)
        print(f"[{idx+1:02d}/{len(problems)}]  difficulty={prob.get('difficulty','?')}  "
              f"id={prob['problem_id']}", flush=True)
        print(f"  PROBLEM: {prob['prompt'][:160]}{'…' if len(prob['prompt'])>160 else ''}",
              flush=True)
        print(f"  ANSWER:  {prob['ground_truth']}", flush=True)

        chunks = _generate_atomic_chunks(
            model=model, tokenizer=tok,
            problem=prob,
            max_chunks=args.max_chunks,
            max_step_tokens=args.max_step_tokens,
            min_step_tokens=args.min_step_tokens,
            device=device,
        )

        # Grade: concatenate all chunk texts
        full_text = "\n".join(c["text"] for c in chunks)
        pred      = extract_answer(full_text)
        reward    = int(pred is not None and answers_equivalent(pred, prob["ground_truth"]))

        # Print each chunk
        for ci, chunk in enumerate(chunks, 1):
            bq = _boundary_quality(chunk["text"])
            print(f"\n  ── Step {ci}  ({chunk['n_tokens']} tokens)  "
                  f"stopped_by={chunk['stopped_by']}  boundary={bq}", flush=True)
            snippet = chunk["text"][:500] + ("…" if len(chunk["text"]) > 500 else "")
            print(_wrap(snippet), flush=True)

        print(f"\n  ── Grade", flush=True)
        print(f"     n_chunks:  {len(chunks)}", flush=True)
        print(f"     predicted: {pred!r}", flush=True)
        print(f"     expected:  {prob['ground_truth']!r}", flush=True)
        print(f"     reward:    {'✅' if reward else '❌'}", flush=True)
        print(f"     total_tokens: {sum(c['n_tokens'] for c in chunks)}", flush=True)
        stop_reason = chunks[-1]["stopped_by"] if chunks else "none"
        print(f"     final stop: {stop_reason}", flush=True)

        all_records.append({
            "problem_id":   prob["problem_id"],
            "difficulty":   prob.get("difficulty"),
            "prompt":       prob["prompt"],
            "ground_truth": prob["ground_truth"],
            "n_chunks":     len(chunks),
            "chunks":       chunks,
            "predicted":    pred,
            "reward":       reward,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    n_correct     = sum(r["reward"]   for r in all_records)
    n_chunks_list = [r["n_chunks"]    for r in all_records]
    stop_reasons  = [r["chunks"][-1]["stopped_by"] for r in all_records if r["chunks"]]

    print(f"\n{'═'*80}", flush=True)
    print(f"SUMMARY  —  {n_correct}/{len(all_records)} correct  "
          f"({100*n_correct/max(len(all_records),1):.1f}%)", flush=True)

    # Per-difficulty breakdown
    by_diff: dict[int, list[int]] = {}
    for r in all_records:
        by_diff.setdefault(r["difficulty"], []).append(r["reward"])
    for d in sorted(by_diff):
        rw = by_diff[d]
        print(f"  L{d}: {sum(rw)}/{len(rw)}", flush=True)

    print(f"\nChunk count distribution:", flush=True)
    from collections import Counter
    for n, cnt in sorted(Counter(n_chunks_list).items()):
        print(f"  {n} chunks: {cnt} problems", flush=True)
    print(f"  avg: {sum(n_chunks_list)/max(len(n_chunks_list),1):.1f} chunks/problem",
          flush=True)

    print(f"\nFinal-chunk stop reason distribution:", flush=True)
    for reason, cnt in sorted(Counter(stop_reasons).items()):
        print(f"  {reason}: {cnt}", flush=True)
    print(flush=True)
    print("What to look for:", flush=True)
    print("  boxed    = model found answer naturally    → GOOD", flush=True)
    print("  eos      = model stopped at step end       → GOOD (if n_chunks > 1)", flush=True)
    print("  max_tok  = model ran to token limit        → model ignoring stop instruction", flush=True)
    print("  max_chks = never produced \\boxed{}         → model can't solve or ran away", flush=True)

    # Per-chunk step token stats
    all_step_tokens = [c["n_tokens"] for r in all_records for c in r["chunks"]]
    if all_step_tokens:
        print(f"\nStep token stats across all chunks:", flush=True)
        print(f"  min={min(all_step_tokens)}  "
              f"max={max(all_step_tokens)}  "
              f"mean={sum(all_step_tokens)/len(all_step_tokens):.0f}", flush=True)
        at_max = sum(1 for t in all_step_tokens if t >= args.max_step_tokens)
        print(f"  {at_max}/{len(all_step_tokens)} steps hit the max_step_tokens cap "
              f"(model may be ignoring stop instruction)", flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\nRecords written to {out_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--step", choices=["traces"], required=True,
                   help="Which assumption check to run.")

    # data
    p.add_argument("--n",      type=int,  default=20)
    p.add_argument("--seed",   type=int,  default=42)
    p.add_argument("--pool",   type=Path, nargs="+", default=[DEFAULT_POOL],
                   help="One or more JSONL pool files (merged before sampling). "
                        "Default: data/math_easy_pool.jsonl")
    p.add_argument("--levels", type=int,  nargs="+", default=None,
                   help="Difficulty levels to sample (e.g. --levels 3 4). "
                        "Default: all levels found in the pool.")

    # model
    p.add_argument("--model",    default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)

    # atomic generation
    p.add_argument("--max-chunks",    type=int, default=12, dest="max_chunks",
                   help="Max reasoning steps per problem before giving up.")
    p.add_argument("--max-step-tok",  type=int, default=512, dest="max_step_tokens",
                   help="Max tokens per reasoning step.")
    p.add_argument("--min-step-tok",  type=int, default=10,  dest="min_step_tokens",
                   help="Min tokens per reasoning step (prevents immediate EOS).")

    # output
    p.add_argument("--out", type=str, default=None,
                   help="Optional JSON output path.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.step == "traces":
        run_traces(args)
    else:
        print(f"Unknown step: {args.step}")
        sys.exit(1)


if __name__ == "__main__":
    main()

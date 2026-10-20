"""Model loading + batched scoring helpers shared by every script."""

import torch
from transformer_lens import HookedTransformer

MODELS = {
    "Llama-3.1-8B":    "meta-llama/Llama-3.1-8B",
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B":      "Qwen/Qwen2.5-7B",
    "pythia-6.9b":     "EleutherAI/pythia-6.9b",
    "gpt2-xl":         "gpt2-xl",
}


def load(model_key, device="cuda:0"):
    m = HookedTransformer.from_pretrained_no_processing(
        MODELS[model_key], device=device, dtype=torch.bfloat16)
    m.eval()
    # Every prompt starts with a BOS position (position 0 = the sink; prefix = 1..p_last). TransformerLens
    # turns this off for Qwen2 and its tokenizer adds none, so Qwen gets '<|endoftext|>' prepended, as
    # GPT-2 and Pythia do. No-op for the models whose default is already True.
    m.cfg.default_prepend_bos = True
    m.tokenizer.padding_side = "right"
    if m.tokenizer.pad_token is None:
        m.tokenizer.pad_token = m.tokenizer.eos_token
    return m


def true_lengths(toks_model, toks):
    """Number of real tokens per right-padded row.

    Counting non-pad tokens is WRONG when bos_token_id == pad_token_id (pythia:
    both are 0), because the prepended BOS is then counted as padding and every
    row is read one position early. Scanning from the right is exact in all cases.
    """
    pad = toks_model.tokenizer.pad_token_id
    is_real = (toks != pad)
    n_trailing_pad = (is_real.flip(1).float().argmax(1))     # first real from the right
    all_pad = ~is_real.any(1)
    n_trailing_pad = torch.where(all_pad, torch.zeros_like(n_trailing_pad), n_trailing_pad)
    return toks.shape[1] - n_trailing_pad


def spaced_piece(tokenizer, i):
    """decode([i]) with the word-initial space kept. SentencePiece tokenizers (Mistral) drop the
    leading space of a '▁word' piece when it is decoded alone; byte-level BPE (GPT-2, Pythia, Llama-3,
    Qwen) keeps it as ' word', so for those models this is exactly decode([i])."""
    s = tokenizer.decode([i])
    if not s.startswith(" "):
        t = tokenizer.convert_ids_to_tokens(i)
        if isinstance(t, str) and t.startswith("▁"):
            s = " " + t[1:]
    return s


def answer_ids(model, answer):
    """Token ids of ' answer' (leading space = the continuation form)."""
    return model.tokenizer.encode(" " + answer, add_special_tokens=False)


@torch.no_grad()
def last_token_logits(model, prompts, batch=32):
    """-> [n, vocab] float32 logits at each prompt's final real token."""
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        toks = model.to_tokens(chunk)                      # BOS-prefixed, right-padded
        lens = true_lengths(model, toks)
        logits = model(toks, return_type="logits")
        idx = torch.arange(len(chunk), device=logits.device)
        out.append(logits[idx, lens - 1, :].float().cpu())
        del logits, toks
    return torch.cat(out, 0)


@torch.no_grad()
def answer_logprob(model, prompts, answers, batch=16):
    """Teacher-forced sum of log P over all tokens of ' answer'. -> list[float]"""
    res = []
    for i in range(0, len(prompts), batch):
        p_chunk, a_chunk = prompts[i:i + batch], answers[i:i + batch]
        full = [p + " " + a for p, a in zip(p_chunk, a_chunk)]
        toks = model.to_tokens(full)
        lens = true_lengths(model, toks)
        logprobs = torch.log_softmax(model(toks, return_type="logits").float(), dim=-1)
        for j, (p, a) in enumerate(zip(p_chunk, a_chunk)):
            n_ans = len(answer_ids(model, a))
            end = int(lens[j])                     # position after the last answer token
            start = end - n_ans                    # first answer token position
            tot = 0.0
            for t in range(start, end):
                tot += float(logprobs[j, t - 1, toks[j, t]])
            res.append(tot)
        del logprobs, toks
    return res

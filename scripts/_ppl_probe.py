"""Decisive probe: is single-chunk prefill quantization-blind?

Scores one long held-out text with prompt_logprobs under lossless / e4m3 / fp4
and prints NLL/PPL at 6 decimals. If all are bit-identical, prefill does not
read the quantized cache and the benchmark must move to decode-mode scoring.
"""
import argparse
import math
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

TEXT = """The history of computing is a story of steady abstraction. Early
machines were built from vacuum tubes and relays, each one performing a single
logical operation at a time. Programmers wrote in machine code, entering
instructions as raw numbers that the hardware could execute directly. The
introduction of assembly language was a small but profound step because it gave
names to the operations and let the programmer think in symbols rather than
binary patterns. From there high-level languages followed quickly. Fortran
arrived to serve scientists who wanted to write formulas. Cobol arrived to serve
businesses that wanted to describe records and reports. Each new language hid
another layer of machine detail behind a friendlier surface.

The same pattern repeated itself in operating systems. The earliest computers
ran one program at a time and required an operator to load the next job by hand.
Batch systems automated the loading but still ran a single job to completion.
Time-sharing then divided the processor among many users so that each one felt
as though the machine belonged to them alone. This required the hardware to
protect memory, the kernel to schedule processes, and the file system to keep
track of data that survived from one session to the next. None of these ideas
was obvious at the time, and all of them remain with us today in forms that
would be recognizable to the engineers who first built them.

Networks introduced a new set of abstractions. A packet of data travels across
many independent links, each of which may be unreliable, slow, or temporarily
congested. The designers of the Internet chose to keep the network itself
simple and to push complexity to the edges of the system. End hosts are
responsible for retransmitting lost packets, ordering those that arrive out of
sequence, and adapting their sending rate to the capacity of the path. This
decision made the network easy to extend and almost impossible to control from
the center, which is exactly why it grew the way it did.

Artificial intelligence has followed a similar trajectory of abstraction. Early
systems encoded expert knowledge as rules and struggled with the messiness of
real language and vision. Statistical methods replaced the rules with models
that learned from data, and neural networks replaced handcrafted features with
representations that the network discovers for itself. The transformer
architecture then removed the recurrence that had made sequence modeling slow
and instead let every token attend to every other token in parallel. This
change, combined with enormous datasets and specialized hardware, produced the
language models we use today. Each advance hid a lower level of computation
behind a higher level of description.

The practical consequence is that modern programmers rarely touch the hardware
at all. They compose libraries, orchestrate services, and reason about data at a
level far removed from the transistors that ultimately do the work. Yet the
lower layers have not disappeared. They have simply become reliable enough to
ignore most of the time. Understanding them still matters when performance is
critical, when a system behaves unexpectedly, or when a new device changes the
rules that everyone had taken for granted. The abstractions are a ladder, and
every rung rests on the ones below it.

Scientific progress often works the same way. Measurements are gathered in
carefully controlled experiments and then compressed into laws that summarize a
great deal of data in a single equation. Newton did not explain why objects
attract one another; he described how they attract with a precision that
allowed engineers to predict the motion of planets and projectiles. Two
centuries later Einstein showed that the description itself was an
approximation, valid at low speeds and weak fields but not in general. The newer
theory did not make the older one useless; it marked the boundary where the
older one stops being accurate.

Biology offers a different kind of example. Living systems are not designed from
a blueprint so much as accumulated through variation and selection. A mutation
that happens to help an organism reproduce will tend to spread through a
population even if no one understands why it helps. The result is a system of
extraordinary complexity that no single engineer would have chosen, full of
redundancy and leftover features that made sense in a distant ancestor.
Evolution is indifferent to elegance, and this is part of why it is so
difficult to predict.

The study of language reveals yet another layer. Human languages are learned by
children with remarkable speed and uniformity, which suggests that the mind
brings strong prior expectations to the task. Yet the languages themselves
differ enormously in their sounds, their word order, and the categories they
choose to mark. Some mark time on the verb, others mark evidence for how a
speaker learned something, and still others mark the shape of the object being
counted. The common thread is that every human language is learnable by a
child, and every one is expressive enough for the community that uses it.

Taken together these examples point to a general truth. Complex systems are
almost always built in layers, and the layers are almost always imperfect. The
value of an abstraction is not that it hides everything but that it hides the
right things, leaving the essential structure visible. The art of engineering,
and perhaps the art of science as well, is knowing which details belong at
which level and when it is time to look beneath the surface.
"""


def _nll_from_prompt_logprobs(out) -> tuple[float, int]:
    s = 0.0
    n = 0
    for lp in out.prompt_logprobs[1:]:
        if lp is None:
            continue
        for t in lp.values():
            if t.rank == 1:
                s -= t.logprob
                n += 1
                break
    return s, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--precisions", nargs="+", dest="precs",
                    default=["lossless", "e4m3", "fp4"])
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    sp = SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=1)
    for prec in args.precs:
        os.environ["PSPKV_PRECISION"] = prec
        os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
        os.environ.setdefault("PSPKV_FORMAT", "1")
        llm = LLM(
            model=args.model,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.6,
            dtype="float16",
            attention_backend="PSP_KV",
        )
        outs = llm.generate([TEXT], sp)
        s, n = _nll_from_prompt_logprobs(outs[0])
        mean = s / max(n, 1)
        print(f"precision={prec} NLL {mean:.6f} PPL {math.exp(mean):.6f} "
              f"tokens {n}")
        del llm


if __name__ == "__main__":
    main()

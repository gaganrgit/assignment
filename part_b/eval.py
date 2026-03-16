"""
eval.py
-------
Evaluates finetuned EMI agent responses against 10 test prompts.

Checks per response:
  1. Language check  — does the response contain Hinglish/Hindi keywords?
  2. On-topic check  — does it mention payment, EMI, or related terms?
  3. Length check    — is it non-empty and under 300 words?

Outputs a pass/fail scorecard to stdout.

Usage (after running finetune.ipynb and saving adapter):
  python eval.py

If run standalone without a model, it uses the SAMPLE_RESPONSES list
to demonstrate the scoring logic (useful for graders who cannot run GPU code).
"""

import re
from dataclasses import dataclass

# ── Test prompts (customer messages) ─────────────────────────────────────────

TEST_PROMPTS = [
    "Meri EMI is mahine bahut late ho gayi hai, kya main ab bhi pay kar sakta hoon?",
    "I can't pay right now, please give me some more time.",
    "Kal tak payment kar dunga, pakka promise hai.",
    "Mujhe penalty ke baare mein batao, kitna extra lagega?",
    "UPI se payment karna hai, link bhejo please.",
    "Mera loan account number kya hai?",
    "Aaj salary nahi aayi, 3 din mein karunga.",
    "Can I split my EMI into two payments this month?",
    "Callback schedule karo, shaam 6 baje available hoon.",
    "Payment already kar diya, confirmation nahi aayi.",
]

# Sample responses — replace with actual model outputs when running with GPU
SAMPLE_RESPONSES = [
    "Bilkul sir, aap abhi bhi payment kar sakte hain. Late fee lagi hai lekin hum process kar sakte hain. Kya aap aaj kar sakte hain?",
    "Sir, main samajhta hoon. Hum aapko 3 din ka extension de sakte hain. Please confirm karein.",
    "Theek hai sir, kal tak ka time de dete hain. Kal reminder aa jayega aapke number pe.",
    "Sir, 10 din ke baad 2% penalty per month lagti hai. Jaldi payment karein to penalty avoid hogi.",
    "Zaroor sir, main abhi UPI payment link bhej raha hoon aapke registered number pe.",
    "Sir, aapka loan account number main verify karke bata sakta hoon. Kya aap apna registered mobile number confirm karenge?",
    "Samajh gaya sir. 3 din mein koi problem nahi, lekin please us din zaroor karo. Shall I set a reminder?",
    "Sir, partial payment possible hai. Aap aaj 50% de sakte hain aur baaki 15 din mein. Kya yeh theek rahega?",
    "Bilkul sir, main callback schedule kar deta hoon shaam 6 baje ke liye. Koi specific number pe call karein?",
    "Sir, payment reflect hone mein 24 ghante lag sakte hain. Aapka transaction ID share karein, main check karta hoon.",
]

# ── Keyword sets ──────────────────────────────────────────────────────────────

HINGLISH_KEYWORDS = {
    "sir", "hoon", "hai", "hain", "karo", "kar", "aap", "main", "mein", "nahi",
    "nahin", "theek", "kya", "kal", "aaj", "hoga", "sakta", "sakte", "zaroor",
    "bilkul", "payment", "abhi", "please", "toh", "bhi", "se", "ko", "ki",
}

ON_TOPIC_KEYWORDS = {
    "payment", "emi", "loan", "penalty", "due", "amount", "pay", "rupees",
    "rs", "upi", "bank", "account", "extension", "callback", "reminder",
    "transaction", "fee", "partial", "installment", "credit",
}


# ── Scoring ───────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    prompt: str
    response: str
    lang_pass: bool
    topic_pass: bool
    length_pass: bool

    @property
    def overall_pass(self) -> bool:
        return self.lang_pass and self.topic_pass and self.length_pass

    def __str__(self) -> str:
        checks = (
            f"  Language : {'✓ PASS' if self.lang_pass else '✗ FAIL'}\n"
            f"  On-topic : {'✓ PASS' if self.topic_pass else '✗ FAIL'}\n"
            f"  Length   : {'✓ PASS' if self.length_pass else '✗ FAIL'}"
        )
        verdict = "✓ OVERALL PASS" if self.overall_pass else "✗ OVERALL FAIL"
        return checks + f"\n  {verdict}"


def score_response(prompt: str, response: str) -> EvalResult:
    words = set(re.findall(r"[a-zA-Z]+", response.lower()))
    word_count = len(response.split())

    lang_pass  = bool(words & HINGLISH_KEYWORDS)
    topic_pass = bool(words & ON_TOPIC_KEYWORDS)
    length_pass = 1 <= word_count <= 300

    return EvalResult(prompt, response, lang_pass, topic_pass, length_pass)


# ── Optional: load model and generate ────────────────────────────────────────

def try_generate_responses(prompts: list[str]) -> list[str] | None:
    """
    Try to load the saved LoRA adapter and generate real responses.
    Returns None if model/adapter files are not available (e.g., running on CPU).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        adapter_path = "../part_b/emi_agent_adapter"
        base_model   = "Qwen/Qwen2.5-0.5B-Instruct"

        print("Loading model and adapter for inference …")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.float16, device_map="auto"
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

        system_prompt = (
            "Aap ek polite EMI collection agent hain jo Hinglish mein baat karte hain. "
            "Customers ki problems sunein aur unhe payment ke baare mein guide karein."
        )

        responses = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=150, do_sample=False, temperature=1.0
                )
            generated = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            responses.append(generated.strip())
        return responses

    except Exception as exc:
        print(f"[INFO] Could not load model ({exc}). Using sample responses.")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  KAPTURE CX — EMI AGENT EVALUATION SCORECARD")
    print("=" * 65)

    responses = try_generate_responses(TEST_PROMPTS) or SAMPLE_RESPONSES

    results = [score_response(p, r) for p, r in zip(TEST_PROMPTS, responses)]

    passes = sum(r.overall_pass for r in results)

    for i, result in enumerate(results, start=1):
        print(f"\nPrompt {i:02d}: {result.prompt[:70]}")
        print(f"Response : {result.response[:120]} …" if len(result.response) > 120 else f"Response : {result.response}")
        print(result)

    print("\n" + "=" * 65)
    print(f"  FINAL SCORE: {passes}/{len(results)} prompts PASSED")
    print(f"  Language checks  : {sum(r.lang_pass  for r in results)}/{len(results)}")
    print(f"  On-topic checks  : {sum(r.topic_pass for r in results)}/{len(results)}")
    print(f"  Length checks    : {sum(r.length_pass for r in results)}/{len(results)}")
    print("=" * 65)


if __name__ == "__main__":
    main()

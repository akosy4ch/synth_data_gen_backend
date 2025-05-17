import pandas as pd
import asyncio

# === Глобальный кэш для моделей ===
model_cache = {}

def _maybe_compile_model(model):
    try:
        import torch
        if hasattr(torch, "compile"):
            print("🔧 Compiling model...")
            return torch.compile(model)
    except Exception:
        pass
    return model

def _move_to_cuda(model):
    try:
        import torch
        if torch.cuda.is_available():
            return model.to("cuda")
    except Exception:
        pass
    return model

def _warmup_model(model, tokenizer, prompt="Hello world"):
    import torch
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        model.generate(**inputs, max_length=10)

def _generate_from_model(model, tokenizer, prompts, max_length, pad_token_id, temperature=0.9, top_k=50, top_p=0.95, attention_mask=True):
    import torch
    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        generate_kwargs = dict(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=pad_token_id,
            use_cache=True
        )
        if attention_mask and "attention_mask" in inputs:
            generate_kwargs["attention_mask"] = inputs["attention_mask"]
        outputs = model.generate(**generate_kwargs)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

async def generate_text(df, model_type, samples, column):
    return await asyncio.to_thread(_generate_text_sync, df, model_type, samples, column)

def _generate_text_sync(df, model_type, samples, column):
    global model_cache

    texts = df[column].dropna().astype(str).tolist()
    if not texts:
        return pd.DataFrame({column: [""] * samples})

    if model_type == "MARKOV":
        import markovify
        model = markovify.Text(" ".join(texts))
        generated = [model.make_sentence() or "" for _ in range(samples)]
        return postprocess_and_fill(pd.DataFrame({column: generated}), column, samples)

    elif model_type == "FLAN-T5":
        if "FLAN-T5" not in model_cache:
            print(f"🔄 Loading model: FLAN-T5 (small)...")
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
            model = _maybe_compile_model(model)
            model = _move_to_cuda(model)
            model_cache["FLAN-T5"] = (tokenizer, model)
            _warmup_model(model, tokenizer)
        tokenizer, model = model_cache["FLAN-T5"]

        prompts = [f"Paraphrase this: {text}" for text in texts[:samples]]
        decoded = _generate_from_model(
            model, tokenizer, prompts, max_length=64, pad_token_id=tokenizer.eos_token_id
        )
        generated = [d.strip() for d in decoded]
        while len(generated) < samples:
            generated.extend(generated[:samples - len(generated)])
        return postprocess_and_fill(pd.DataFrame({column: generated}), column, samples)


    elif model_type == "GPT-J":
        if "GPT-J" not in model_cache:
            print(f"🔄 Loading model: GPT-J...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            tokenizer.pad_token = tokenizer.eos_token
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    "EleutherAI/gpt-j-6B",
                    device_map="auto",
                    load_in_4bit=True
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
            model = _maybe_compile_model(model)
            model = _move_to_cuda(model)
            model_cache["GPT-J"] = (tokenizer, model)
            _warmup_model(model, tokenizer)
        tokenizer, model = model_cache["GPT-J"]

        prompts = [texts[i % len(texts)][:100] for i in range(samples)]
        decoded = _generate_from_model(
            model, tokenizer, prompts, max_length=150, pad_token_id=tokenizer.eos_token_id,
            temperature=0.9, top_k=40, top_p=0.9
        )
        cleaned = [
            d[len(p):].strip() if d.startswith(p) else d.strip()
            for d, p in zip(decoded, prompts)
        ]
        return postprocess_and_fill(pd.DataFrame({column: cleaned}), column, samples)

    elif model_type == "LLAMA-2-7B":
        if "LLAMA-2-7B" not in model_cache:
            print(f"🔄 Loading model: LLAMA-2-7B...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    "NousResearch/Llama-2-7b-chat-hf",
                    device_map="auto",
                    load_in_4bit=True
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    "NousResearch/Llama-2-7b-chat-hf",
                    device_map="auto"
                )
            model = _maybe_compile_model(model)
            model = _move_to_cuda(model)
            model_cache["LLAMA-2-7B"] = (tokenizer, model)
            _warmup_model(model, tokenizer)
        tokenizer, model = model_cache["LLAMA-2-7B"]

        prompts = [texts[i % len(texts)][:100] for i in range(samples)]
        decoded = _generate_from_model(
            model, tokenizer, prompts, max_length=150, pad_token_id=tokenizer.eos_token_id,
            temperature=0.7, top_k=40, top_p=0.9
        )
        cleaned = [
            d[len(p):].strip() if d.startswith(p) else d.strip()
            for d, p in zip(decoded, prompts)
        ]
        return postprocess_and_fill(pd.DataFrame({column: cleaned}), column, samples)

    elif model_type == "DEEPSEEK":
        if "DEEPSEEK" not in model_cache:
            print(f"🔄 Loading model: DEEPSEEK...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/DeepSeek-R1",
                    trust_remote_code=True,
                    device_map="auto",
                    load_in_4bit=True
                )
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    "deepseek-ai/DeepSeek-R1",
                    trust_remote_code=True
                )
            model = _maybe_compile_model(model)
            model = _move_to_cuda(model)
            model_cache["DEEPSEEK"] = (tokenizer, model)
            _warmup_model(model, tokenizer)
        tokenizer, model = model_cache["DEEPSEEK"]

        prompts = [texts[i % len(texts)].strip() + "\nContinue:" for i in range(samples)]
        decoded = _generate_from_model(
            model, tokenizer, prompts, max_length=150, pad_token_id=tokenizer.eos_token_id
        )
        cleaned = [
            d[len(p):].strip() if d.startswith(p) else d.strip()
            for d, p in zip(decoded, prompts)
        ]
        return postprocess_and_fill(pd.DataFrame({column: cleaned}), column, samples)

    elif model_type == "OPENELM":
        if "OPENELM" not in model_cache:
            print(f"🔄 Loading OpenELM-1_1B-Instruct model...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            tokenizer = AutoTokenizer.from_pretrained("apple/OpenELM-1_1B-Instruct", trust_remote_code=True)

            try:
                # Attempt to load with GPU and float16
                model = AutoModelForCausalLM.from_pretrained(
                    "apple/OpenELM-1_1B-Instruct",
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            except Exception:
                # Fallback to CPU if GPU or float16 is not available
                model = AutoModelForCausalLM.from_pretrained(
                    "apple/OpenELM-1_1B-Instruct",
                    trust_remote_code=True
                )

            model = _maybe_compile_model(model)
            model = _move_to_cuda(model)  # this will do nothing if no GPU is present
            model_cache["OPENELM"] = (tokenizer, model)
            _warmup_model(model, tokenizer)

        tokenizer, model = model_cache["OPENELM"]

        prompts = [texts[i % len(texts)].strip() + "\nAnswer:" for i in range(samples)]

        import torch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]

        # ⚙️ Custom attention mask (every 2nd token masked out as example)
        attention_mask = input_ids.clone().detach()
        attention_mask[:, :] = 1
        attention_mask[:, ::2] = 0  # mask out every other token

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=150,
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_k=40,
                top_p=0.95,
                do_sample=True
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = [
            d[len(p):].strip() if d.startswith(p) else d.strip()
            for d, p in zip(decoded, prompts)
        ]
        return postprocess_and_fill(pd.DataFrame({column: cleaned}), column, samples)

    elif model_type == "DISTIL-CMLM":
        if "DISTIL-CMLM" not in model_cache:
            print(f"🔄 Loading model: DISTIL-CMLM...")
            from transformers import AutoTokenizer, pipeline
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            pipe = pipeline("fill-mask", model="distilbert-base-uncased")
            model_cache["DISTIL-CMLM"] = (tokenizer, pipe)

        tokenizer, pipe = model_cache["DISTIL-CMLM"]

        generated = []
        for i in range(samples):
            seed = texts[i % len(texts)]
            seed = " ".join(seed.split()[:5]) + " [MASK]"
            try:
                result = pipe(seed)
                if isinstance(result, list) and result:
                    filled = result[0]["sequence"]
                    generated.append(filled)
                else:
                    generated.append(seed)
            except:
                generated.append(seed)

        return postprocess_and_fill(pd.DataFrame({column: generated}), column, samples)

    else:
        raise ValueError(f"Unsupported text model: {model_type}")

def postprocess_and_fill(df, column, samples):
    df[column] = df[column].fillna("").astype(str)
    df = df[df[column].str.strip() != ""]
    while len(df) < samples:
        df = pd.concat([df, df]).head(samples)
    return df.reset_index(drop=True)

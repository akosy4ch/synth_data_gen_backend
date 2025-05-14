import pandas as pd
import asyncio

# === Глобальный кэш для моделей ===
model_cache = {}

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
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
            model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
            model_cache["FLAN-T5"] = (tokenizer, model)

        tokenizer, model = model_cache["FLAN-T5"]

        prompts = [f"Paraphrase this: {text}" for text in texts[:samples]]
        generated = []

        for prompt in prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(
                input_ids,
                max_length=64,
                temperature=0.9,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated.append(gen)

        while len(generated) < samples:
            generated.extend(generated[:samples - len(generated)])
        return postprocess_and_fill(pd.DataFrame({column: generated}), column, samples)

    elif model_type == "GPT-J":
        if "GPT-J" not in model_cache:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
            model_cache["GPT-J"] = (tokenizer, model)

        tokenizer, model = model_cache["GPT-J"]

        generated = []
        for i in range(samples):
            prompt = texts[i % len(texts)][:100]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=150,
                temperature=0.9,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            cleaned = decoded.replace(prompt, "").strip()
            generated.append(cleaned)

        return postprocess_and_fill(pd.DataFrame({column: generated}), column, samples)

    elif model_type == "LLAMA-2-7B":
        if "LLAMA-2-7B" not in model_cache:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "NousResearch/Llama-2-7b-chat-hf",
                device_map="auto",
                load_in_4bit=True
            )
            model_cache["LLAMA-2-7B"] = (tokenizer, model)

        tokenizer, model = model_cache["LLAMA-2-7B"]

        generated = []
        for i in range(samples):
            prompt = texts[i % len(texts)][:100]
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=150,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            cleaned = decoded.replace(prompt, "").strip()
            generated.append(cleaned)

        return postprocess_and_fill(pd.DataFrame({column: generated}), column, samples)

    elif model_type == "DEEPSEEK":
        if "DEEPSEEK" not in model_cache:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
            model_cache["DEEPSEEK"] = (tokenizer, model)

        tokenizer, model = model_cache["DEEPSEEK"]

        generated = []
        for i in range(samples):
            prompt = texts[i % len(texts)].strip() + "\nContinue:"
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            output = model.generate(
                inputs,
                max_length=150,
                temperature=0.9,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            cleaned = decoded.replace(prompt, "").strip()
            generated.append(cleaned)

        return postprocess_and_fill(pd.DataFrame({column: generated}), column, samples)

    elif model_type == "DISTIL-CMLM":
        if "DISTIL-CMLM" not in model_cache:
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

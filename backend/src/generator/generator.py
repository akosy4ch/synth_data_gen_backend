import pandas as pd
from pandas.api.types import is_string_dtype
from .numeric_generator import generate_numeric
from .utils import validate_model_for_columns
import logging

# Map model name to its corresponding module
from .text_models import (
    flan_t5,
    gpt_j,
    llama,
    deepseek,
    openelm,
    distil_cmlm,
    markov_gen,
)

TEXT_MODEL_ROUTER = {
    "flan-t5": flan_t5.generate_text,
    "gpt-j": gpt_j.generate_text,
    "llama-3.2": llama.generate_text,
    "deepseek": deepseek.generate_text,
    "openelm": openelm.generate_text,
    "distil-cmlm": distil_cmlm.generate_text,
    "markov": markov_gen.generate_text,
}

async def generate_synthetic(df, model_type, samples, task, column):
    model_type = validate_model_for_columns(task, model_type, df[[column]])

    if task == "text" and is_string_dtype(df[column]):
        from asyncio import get_running_loop
        loop = get_running_loop()
        prompt_list = df[column].dropna().astype(str).tolist()

        def batch():
            generator = TEXT_MODEL_ROUTER.get(model_type.lower())
            if not generator:
                raise ValueError(f"Unsupported text model: {model_type}")
            results = []
            for prompt in prompt_list:
                try:
                    df_out = generator(
                                prompt=prompt,
                                model_name=model_type,
                                num_samples=1,
                                df=df,
                                column=column
                            )
                    if isinstance(df_out, pd.DataFrame) and "generated" in df_out.columns:
                        results.extend(df_out['generated'].tolist())
                    elif isinstance(df_out, list):
                        results.extend(df_out)
                    else:
                        results.append(str(df_out))
                except Exception as e:
                    logging.warning(f"⚠️ Generation failed on prompt: {e}")
                    results.append("")
            return pd.DataFrame({column: results[:samples]})



        result_df = await loop.run_in_executor(None, batch)


        return result_df

    elif task == "numeric":
        return await generate_numeric(df[[column]], model_type, samples)

    raise ValueError(f"Unsupported task type: {task}")

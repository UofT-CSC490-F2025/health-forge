from typing import List, Tuple
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np

class BioMistralVectorTagger:

    model = None
    pipeline = None
    vector_definitions: np.ndarray = None
    encoder = None
    bert_tokenizer = None
    def __init__(self, vector_definitions: List[str]):

        print("--------------------------------------------")
        print(f"Checking for LLM CUDA availability")
        print(f"Cuda Availability: {torch.cuda.is_available()}")
        print(f"Cuda Device Name : {torch.cuda.get_device_name(0)}")
        print(f"Cuda Version: {torch.version.cuda}")
        print("--------------------------------------------\n\n")

        self.vector_definitions = vector_definitions

        BIOMISTRAL = "BioMistral/BioMistral-7B"

        self.llm_tokenizer = AutoTokenizer.from_pretrained(BIOMISTRAL)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            BIOMISTRAL,
            torch_dtype=torch.float16,  
            device_map="auto",         
        )

        
        llm_pipeline = pipeline("text-generation", model=self.model, tokenizer=self.llm_tokenizer)

        print("Model device: ")
        print(next(self.model.parameters()).device)

        print("--------------------------------------------")
        print("Executing test inference...")

        messages = [{"role": "user", "content": "Who are you?"},]  
        print(f"Prompt: {messages}")

        prompt = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )     

        out = llm_pipeline(
            prompt,
            max_new_tokens=64,
            return_full_text=False
        )

        print(f"Test Output: {out[0]['generated_text']}")
        print("--------------------------------------------\n\n")


        print("Loading bert encoder...")
        BERT = "bert-base-uncased"  # or your BioBERT/ClinicalBERT
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT)
        model = AutoModel.from_pretrained(BERT)
        self.encoder = model
        self.bert_tokenizer = bert_tokenizer
        print("Finished loading encoder")
        

    def encode_text(self, text_batch: List[str]) -> np.ndarray:
        encoded = self.bert_tokenizer(
            text_batch,
            padding=True,          # pad to longest in the batch
            truncation=True,       # cut off if longer than max_length
            max_length=128,        # or whatever you want
            return_tensors="pt",
        ).to(self.encoder.device)

        outputs = self.encoder(**encoded)           # (batch, seq_len, hidden_size)
        last_hidden = outputs.last_hidden_state

        # Example: mean pooling over tokens
        mask = encoded["attention_mask"].unsqueeze(-1).type_as(last_hidden)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = summed / counts                # (batch, hidden_size)

        return emb.detach().cpu().numpy()
    
    def format_vector_full(self, vec):
        txt = ""
        gender_value = int(vec[0])
        gender_str = "male" if gender_value == 1 else "female"
        txt += f"- gender: {gender_str}\n"

        for i, v in enumerate(vec[1:], start=1):
            if float(v) != 0:
                if float(v) == 1.0:
                    txt += f"- {self.vector_definitions[i]}\n"
                else:
                    txt += f"- {self.vector_definitions[i]}: {float(v)}\n"

        return txt.strip()

    
    def tag_vectors(self, vector_batch: np.ndarray) -> List[List]:
        """
        vector_batch: shape (batch_size, n_features)
        returns: list of generated tag strings, length = batch_size
        """

        prompts: List[str] = []
        messages = []
        for vector in vector_batch:
            # Only look at non-zero entries
            vector_mapping = self.format_vector_full(vector)
            
            messages = [
                {
                    "role": "user",
                    "content": f"""
You are a clinical summarization assistant.

You will be given a sparse vector representing one patient's electronic health record.
The vector has {len(vector)} features and consists of:
- patient demographic information
- total number of hospital admissions
- a multi-hot encoding of truncated ICD-10 codes.

Below are all NON-ZERO feature names and their values.

TASK:
- Write EXACTLY ONE sentence in natural language.
- Summarize the patient clinically (age, sex if present, major diagnoses, key comorbidities).
- Do NOT output raw feature names or numeric values like "1.0".
- Do NOT output code-like fragments (e.g. "something disease: 1.0").
- Do NOT output only numbers; always write a full sentence.

NON-ZERO FEATURES:
{vector_mapping}
            """
                }
            ]

            # Use the chat template to build the actual prompt string
            prompt = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        # -----------------------------
        # Batched tokenization
        # -----------------------------
        enc = self.llm_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=8192,          # adjust if needed
            return_tensors="pt",
        )

        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # -----------------------------
        # Batched generation
        # -----------------------------
        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=True,         
                num_beams=1,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )

        # -----------------------------
        # Strip the prompt, keep only generated text
        # (equivalent to pipeline(..., return_full_text=False))
        # -----------------------------
        input_len = enc["input_ids"].shape[1]          # same for all due to padding
        gen_only_ids = output_ids[:, input_len:]       # (batch, generated_len)

        texts = self.llm_tokenizer.batch_decode(
            gen_only_ids,
            skip_special_tokens=True,
        )

        # Clean up whitespace
        texts = [t.strip() for t in texts]

        for i in range(0, len(texts)):
            print(f"PROMPT: {prompts[i]}")
            print(f"RESPONSE: {texts[i]}")

        return [[vector_batch[i], texts[i]] for i in range(0, len(texts))]
    
    


        
if __name__== '__main__':
    bio = BioMistralVectorTagger(np.array(['age']))
    res = bio.tag_vectors(np.array([[69.0], [67.0], [7.0]]))
    print(res)
    print(bio.encode_text([res[i][1] for i in range(0, len(res))]))


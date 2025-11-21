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
    


    def tag_vectors(self, vector_batch: np.ndarray) -> List[List]:
        """
        vector_batch: shape (batch_size, n_features)
        returns: list of generated tag strings, length = batch_size
        """

        prompts: List[str] = []
        messages = []
        for vector in vector_batch:
            # Only look at non-zero entries

            lines = []
            for i in range(0, len(vector)):
                if self.vector_definitions[i] == 'isMale' or vector[i] > 0:
                    lines.append(f"{self.vector_definitions[i]}: {vector[i]}")
        
            vector_mapping = "\n".join(lines)
            
            messages = [
                {
                    "role": "user",
                    "content": f"""Your task is to provide a very short description of a vector representing a patient's electronic health record.
                    The vector has {len(vector)} features. It consists of:
                    - patient demographic information
                    - total number of hospital admissions
                    - a multi-hot encoding of truncated ICD-10 codes the patient has been diagnosed with.

                    Below are all non-zero feature names and their values.
                    Please describe the patient in a concise clinical summary.

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
            max_length=512,          # adjust if needed
            return_tensors="pt",
        )

        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # -----------------------------
        # Batched generation
        # -----------------------------
        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,           # deterministic; set True if you want sampling
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

        return [[vector_batch[i], texts[i]] for i in range(0, len(texts))]



        
if __name__== '__main__':
    bio = BioMistralVectorTagger(np.array(['age']))
    res = bio.tag_vectors(np.array([[69.0], [67.0], [7.0]]))
    print(res)
    print(bio.encode_text([res[i][1] for i in range(0, len(res))]))


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
    
    def format_vector_full(self, vec): # Testable
        txt = "Patient Facts: \n"

        gender_value = int(vec[0])
        gender_str = "male" if gender_value == 1 else "female"
        txt += f"- gender: {gender_str}\n"

        age_value = vec[1]
        txt += f"- age: {age_value}\n"

        alive_value = int(vec[2])
        alive_str = "has deceased" if alive_value == 1 else "is alive"
        txt += f"- {alive_str}\n"

        txt += "- marital status: "
        for i, v in enumerate(vec[4:9], start=4):
            if float(v) == 1.0:
                    txt += f"{self.vector_definitions[i]}"
        txt += "\n"
        txt += "- ethnicity(s): "
        for i, v in enumerate(vec[9:42], start=9):
            if float(v) == 1.0:
                    txt += f"{self.vector_definitions[i]} "            
        txt += "\n\n"

        txt += "Patient Past Diagnoses: \n"
        for i, v in enumerate(vec[42:], start=42):
            if float(v) == 1.0:
                txt += f"- {self.vector_definitions[i]}\n"

        if (vec[42:] == 0.0).all():
             txt += "- Patient has not been diagnosed with any conditions." 

        return txt.strip()

    
    def tag_vectors(self, vector_batch: np.ndarray) -> List[List]: #Test shape
        """
        vector_batch: shape (batch_size, n_features)
        returns: list of [vector, summary] pairs, length = batch_size
        """

        # Will store summaries in order (some from templates, some from LLM)
        summaries: List[str] = []

        # Store prompts only for the samples that *require* LLM inference
        llm_prompts: List[str] = []
        llm_indices: List[int] = []   # map LLM outputs back to original patient index

        # --- STEP 1: Create prompts OR handle "no diagnoses" directly ---
        for idx, vector in enumerate(vector_batch):

            # Build patient text block
            vector_mapping = self.format_vector_full(vector)

            # Detect zero diagnoses
            has_no_dx = (vector[42:] == 0.0).all()

            if has_no_dx:
                # -----------------------------------------
                # CASE A — Skip LLM entirely for this patient
                # -----------------------------------------
                age = int(vector[1])
                gender_value = int(vector[0])
                gender_str = "male" if gender_value == 1 else "female"

                alive_value = int(vector[2])
                alive_str = "has deceased" if alive_value == 1 else "is alive"
        

                marital_status = ""
                for i, v in enumerate(vector[4:9], start=4):
                    if float(v) == 1.0:
                        marital_status += f"{self.vector_definitions[i]}"
                
                ethnic_group = ""
                for i, v in enumerate(vector[9:42], start=9):
                    if float(v) == 1.0:
                        ethnic_group += f"{self.vector_definitions[i]}"            

                # Deterministic summary
                summary = (
                    f"The patient is a {age}-year-old {ethnic_group} {gender_str} with no documented diagnoses. The patient {alive_str}."
                )
                summaries.append(summary)
                continue

            # -----------------------------------------
            # CASE B — Build LLM prompt for this patient
            # -----------------------------------------
            messages = [
                {
                    "role": "user",
                    "content": f"""
    You are a clinical summarization assistant.

    You will be given structured information about a single patient, including
    basic demographics and a list of past diagnoses.

    All of the information in the patient description is factual and should be
    treated as correct. Do not contradict it.

    TASK:
    - Write EXACTLY ONE concise clinical sentence.
    - Summarize the patient in natural language, focusing on:
    - age and gender (if provided),
    - major diagnoses,
    - important comorbidities.
    - You may group related diagnoses into broader clinical concepts.
    - Do NOT invent diagnoses that are not implied by the given information.
    - Do NOT mention raw lists, bullet points, or code-like text.
    - Do NOT explain your reasoning or add extra commentary.
    - Your output MUST be a single sentence ending with a period.

    Here is the patient description:

    {vector_mapping}
    """
                }
            ]

            # Build actual string prompt
            prompt = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            llm_prompts.append(prompt)
            llm_indices.append(idx)
            summaries.append(None)  # placeholder

        # --- EXIT EARLY IF NO LLM WORK NEEDED ---
        if len(llm_prompts) == 0:
            return [[vector_batch[i], summaries[i]] for i in range(len(summaries))]

        # --- STEP 2: Batch tokenization for only LLM samples ---
        enc = self.llm_tokenizer(
            llm_prompts,
            padding=True,
            truncation=True,
            max_length=4096,
            return_tensors="pt",
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # --- STEP 3: Batched generation ---
        with torch.no_grad():
            output_ids = self.model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )

        # Strip prompt from outputs
        input_len = enc["input_ids"].shape[1]
        gen_only = output_ids[:, input_len:]
        decoded = self.llm_tokenizer.batch_decode(
            gen_only, skip_special_tokens=True
        )
        decoded = [t.strip() for t in decoded]

        # --- STEP 4: Insert LLM outputs back into correct positions ---
        for out_text, orig_idx in zip(decoded, llm_indices):
            summaries[orig_idx] = out_text

        # --- STEP 5: Return aligned results ---
        return [[vector_batch[i], summaries[i]] for i in range(len(summaries))]

    
    


        
if __name__== '__main__':
    bio = BioMistralVectorTagger(np.array(['age']))
    res = bio.tag_vectors(np.array([[69.0], [67.0], [7.0]]))
    print(res)
    print(bio.encode_text([res[i][1] for i in range(0, len(res))]))


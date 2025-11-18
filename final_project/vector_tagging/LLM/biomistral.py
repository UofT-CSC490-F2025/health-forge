from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np

class BioMistralVectorTagger:

    model = None
    pipeline = None
    vector_definitions: np.ndarray = None
    def __init__(self, vector_definitions: np.ndarray):

        import torch
        print("--------------------------------------------")
        print(f"Checking for LLM CUDA availability")
        print(f"Cuda Availability: {torch.cuda.is_available()}")
        print(f"Cuda Device Name : {torch.cuda.get_device_name(0)}")
        print(f"Cuda Version: {torch.version.cuda}")
        print("--------------------------------------------\n\n")


        model_id = "BioMistral/BioMistral-7B"

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  
            device_map="auto",         
        )

        
        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=tokenizer)

        print("Model device: ")
        print(next(self.model.parameters()).device)

        print("--------------------------------------------")
        print("Executing test inference...")

        messages = [{"role": "user", "content": "Who are you?"},]  
        print(f"Prompt: {messages}")

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )     

        out = self.pipeline(
            prompt,
            max_new_tokens=64,
            return_full_text=False
        )

        print(f"Test Output: {out[0]["generated_text"]}")
        print("--------------------------------------------\n\n")


        self.vector_definitions = vector_definitions
        

    def tag_vector(self, vector: np.ndarray ):

        vector_mapping = f""
        #Get all non-zero features and describe them for the model
        for i in range(0, len(vector)):
            if vector[i] > 0:
                vector_mapping += f"{self.vector_definitions[i]}:{vector[i]}\n"

        prompt_raw = f"""
            Your task is to provide a very short description of a vector representing a patient's electronic health record. The entire vector is {len(vector)} feature(s) long. 
            It consists of patient demographic information, total number of hospital admissions, and a multi-hot encoding of truncated ICD-10 codes the patient has been diagnosed with. 
            The following is a list of all non-zero feature names of the vector and their associated values. 
            Please describe the vector.

            {vector_mapping}
        """

        print(prompt_raw)
        messages = [{"role": "user", "content": prompt_raw},] 

        prompt = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        vector_tag = self.pipeline(
            prompt,
            max_new_tokens=64,
            return_full_text=False
        )

        return vector_tag[0]["generated_text"]
        


if __name__== '__main__':
    bio = BioMistralVectorTagger(np.array(['age']))
    print(bio.tag_vector(np.array([69.0])))


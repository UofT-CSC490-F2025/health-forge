import torch
import random
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pickle

def create_dummy_samples(sample_size, max_age=100):
    # [male, age, married, dead, diabetes, cancer, depression, schizophrenia]
    samples = []
    for _ in range(sample_size):
        random.randint(0, 1)
        male = random.randint(0, 1)
        age = round(random.randrange(0, max_age) / max_age, 2)
        married = random.randint(0, 1)
        dead = random.randint(0, 1)
        diabetes = random.randint(0, 1)
        cancer = random.randint(0, 1)
        depression = random.randint(0, 1)
        schizophrenia = random.randint(0, 1)
        
        curr_sample = torch.tensor([male, age, married, dead, diabetes, cancer, depression, schizophrenia])
        samples.append(curr_sample)
    
    samples = torch.stack(samples, dim=0)

    return samples


def data_descriptions(samples, max_age=100):
    vec_to_word = {
        0: {0: "is female", 1: "is male"},
        1: lambda x : f"is {int(x * max_age)} years old",
        2: {0: "is not married", 1: "is married"},
        3: {0: "is not dead", 1: "is dead"},
        4: {0: "does not have diabetes", 1: "has diabetes"},
        5: {0: "does not have cancer", 1: "has cancer"},
        6: {0: "does not have depression", 1: "has depression"},
        7: {0: "does not have schizophrenia", 1: "has schizophrenia"},
    }
    
    B, D = samples.shape
    descriptions = []
    for i in range(B):
        curr_sample = samples[i]
        curr_desc = []
        for j in range(D):
            if isinstance(vec_to_word[j], dict):
                prop = curr_sample[j].item()
                curr_desc.append(vec_to_word[j][prop])
            else:
                curr_desc.append(vec_to_word[j](curr_sample[j]))
        descriptions.append(curr_desc)
    
    # descriptions = torch.concat(descriptions)

    return descriptions


def create_samples_and_desc(num_samples, max_age=100):
    samples = create_dummy_samples(num_samples, max_age)
    descriptions = data_descriptions(samples, max_age)

    return samples, descriptions



def create_llm_descs(descs):
        
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    messages = []
    for d in descs:
        messages.append([
            {"role": "user", 
             "content": f"Write a concise, direct english description with no filler of a person with the following properties: {', '.join(d)}"}
        ]) 

    results = pipe(messages)
    # print(results)
    # for i in range(len(results)):
    #     print(results[i][0]["generated_text"][1]["content"])
    
    llm_descs = [results[i][0]["generated_text"][1]["content"] for i in range(len(results))]
    return llm_descs


def get_text_embeds(descs):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(descs)
    return embeddings



if __name__ == "__main__":
    samples, descs = create_samples_and_desc(100, max_age=100)

    llm_descs = create_llm_descs(descs)
    text_embeds = get_text_embeds(llm_descs)

    # print(samples)
    # print(llm_descs)
    # print(text_embeds)
    data = {"samples": samples,
            "descs": descs,
            "llm_descs": llm_descs,
            "text_embeds": text_embeds}
    
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)
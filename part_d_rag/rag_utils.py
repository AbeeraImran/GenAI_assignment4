import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

RETRIEVER = 'sentence-transformers/all-MiniLM-L6-v2'
GENERATOR = 'Qwen/Qwen2.5-1.5B-Instruct'

def load_retriever(device):
    tok = AutoTokenizer.from_pretrained(RETRIEVER)
    model = AutoModel.from_pretrained(RETRIEVER).to(device)
    model.eval()
    return tok, model

def load_generator(device):
    tok = AutoTokenizer.from_pretrained(GENERATOR)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR).half().to(device)
    model.eval()
    return tok, model

def chunk_text(text, size=150, overlap=30):
    words = text.split()
    step = size - overlap
    chunks = []
    start = 0
    while True:
        chunks.append(' '.join(words[start:start + size]))
        if start + size >= len(words):
            break
        start += step
    return chunks

def embed(texts, tok, model, device, batch_size=128):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=256, return_tensors='pt').to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        mean = summed / counts
        mean = mean / mean.norm(dim=1, keepdim=True)
        out.append(mean.cpu())
    return torch.cat(out)

def search_docs(q_emb, emb, chunk_doc, doc_ids, k):
    scores = emb @ q_emb
    best = torch.full((len(doc_ids),), -1e9, device=emb.device)
    best = best.scatter_reduce(0, chunk_doc, scores, reduce='amax')
    top = best.topk(k).indices.tolist()
    return [doc_ids[i] for i in top]

def search_chunks(q_emb, emb, k):
    scores = emb @ q_emb
    return scores.topk(k).indices.tolist()

def build_messages(question, passages):
    text = ''
    for i, p in enumerate(passages):
        text += '[' + str(i + 1) + '] ' + p + '\n\n'
    system = ('You are a biomedical research assistant. Answer the question using only the numbered passages. '
              'Cite the passage numbers in square brackets, for example [1]. '
              'If the passages do not contain the answer, say exactly: I do not know based on the passages.')
    user = 'Passages:\n\n' + text + 'Question: ' + question
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

def generate_answer(question, passages, tok, model, device):
    messages = build_messages(question, passages)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()

def retrieval_metrics(rankings, relevants, k):
    hit = 0
    recall = 0
    for ranked, rel in zip(rankings, relevants):
        rel = set(rel)
        found = len(set(ranked[:k]) & rel)
        if found >= 1:
            hit += 1
        recall += found / min(k, len(rel))
    n = len(rankings)
    return hit / n, recall / n

def mrr10(rankings, relevants):
    total = 0
    for ranked, rel in zip(rankings, relevants):
        rel = set(rel)
        for pos, d in enumerate(ranked[:10]):
            if d in rel:
                total += 1 / (pos + 1)
                break
    return total / len(rankings)

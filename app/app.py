import os
import sys
import json
import numpy as np
import torch
import gradio as gr
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for folder in ['starter', 'part_a_mae', 'part_b_vae', 'part_c_diffusion', 'part_d_rag']:
    sys.path.append(os.path.join(ROOT, folder))

from eurosat import load_eurosat_split, patchify, unpatchify, CLASS_NAMES
from mae_model import MAE
from vae_model import VAE
from ddpm import make_unet, make_schedule, sample
from rag_utils import load_retriever, load_generator, embed, search_chunks, generate_answer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

NOTICE = 'For educational use only. Not medical advice.'

def to_image(t, scale):
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    h, w = arr.shape[0], arr.shape[1]
    return Image.fromarray(arr).resize((w * scale, h * scale), Image.NEAREST)

def make_grid(imgs, ncols):
    n = imgs.shape[0]
    rows = (n + ncols - 1) // ncols
    canvas = torch.zeros(1, rows * 28, ncols * 28)
    for i in range(n):
        r = i // ncols
        c = i % ncols
        canvas[:, r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = imgs[i].cpu()
    return canvas

print('loading MAE')
mae = MAE().to(device)
mae.load_state_dict(torch.load(os.path.join(ROOT, 'part_a_mae', 'mae.pt'), map_location=device))
mae.eval()
clf = torch.nn.Linear(192, 10).to(device)
clf.load_state_dict(torch.load(os.path.join(ROOT, 'part_a_mae', 'clf.pt'), map_location=device))
x_test, y_test = load_eurosat_split('test')

print('loading VAE')
vae = VAE(16).to(device)
vae.load_state_dict(torch.load(os.path.join(ROOT, 'part_b_vae', 'vae.pt'), map_location=device))
vae.eval()
examples = np.load(os.path.join(ROOT, 'app', 'uhat_class_examples.npz'))
class_names = [str(n) for n in examples['names']]
class_images = torch.from_numpy(examples['images'])

print('loading diffusion')
unet = make_unet().to(device)
ck = torch.load(os.path.join(ROOT, 'part_c_diffusion', 'ddpm.pt'), map_location=device, weights_only=False)
unet.load_state_dict(ck['unet'])
unet.eval()
betas, alphas, alpha_bar = make_schedule(device)

print('loading RAG')
ret_tok, ret_model = load_retriever(device)
gen_tok, gen_model = load_generator(device)
emb = torch.load(os.path.join(ROOT, 'part_d_rag', 'chunk_embeddings.pt')).to(device)
with open(os.path.join(ROOT, 'part_d_rag', 'chunks.json')) as f:
    saved = json.load(f)
chunks = saved['texts']
chunk_doc_ids = saved['doc_ids']

def run_mae(index):
    index = int(index)
    img = x_test[index:index + 1].to(device)
    patches = patchify(img)
    with torch.no_grad():
        pred, mask = mae(patches)
        label = clf(mae.features(patches)).argmax(dim=1).item()
    m = mask.unsqueeze(-1)
    masked_in = patches * (1 - m) + 0.5 * m
    recon = patches * (1 - m) + pred * m
    text = 'Predicted: ' + CLASS_NAMES[label] + '\nTrue: ' + CLASS_NAMES[y_test[index].item()]
    return (to_image(unpatchify(masked_in)[0], 4),
            to_image(unpatchify(recon)[0], 4),
            to_image(img[0], 4),
            text)

def vae_random():
    z = torch.randn(32, 16, device=device)
    with torch.no_grad():
        out = torch.sigmoid(vae.decode(z))
    return to_image(make_grid(out, 8), 4)

def interpolate(name1, name2, alpha):
    x1 = class_images[class_names.index(name1)].unsqueeze(0).to(device)
    x2 = class_images[class_names.index(name2)].unsqueeze(0).to(device)
    with torch.no_grad():
        m1, _ = vae.encode(x1)
        m2, _ = vae.encode(x2)
        z = m1 + alpha * (m2 - m1)
        out = torch.sigmoid(vae.decode(z))
    row = torch.cat([x1.cpu(), out.cpu(), x2.cpu()])
    return to_image(make_grid(row, 3), 6)

def run_diffusion(name):
    label = class_names.index(name)
    labels = torch.full((8,), label, device=device, dtype=torch.long)
    out, _ = sample(unet, labels, betas, alphas, alpha_bar)
    out = (out + 1) / 2
    return to_image(make_grid(out, 4), 6)

def run_rag(question):
    question = question.strip()
    if question == '':
        return '', '', NOTICE
    q_emb = embed([question], ret_tok, ret_model, device)[0].to(device)
    idx = search_chunks(q_emb, emb, 5)
    passages = [chunks[j] for j in idx]
    docs = [chunk_doc_ids[j] for j in idx]
    answer = generate_answer(question, passages, gen_tok, gen_model, device)
    shown = ''
    for n in range(5):
        shown += '[' + str(n + 1) + '] abstract id ' + str(docs[n]) + '\n' + passages[n] + '\n\n'
    return answer, shown, NOTICE

with gr.Blocks(title='Generative AI Assignment 4') as demo:
    gr.Markdown('# Generative AI Assignment 4')

    with gr.Tab('MAE'):
        gr.Markdown('Masked autoencoder on EuroSAT. Pick a test tile.')
        tile = gr.Slider(0, len(x_test) - 1, value=0, step=1, label='Test tile number')
        mae_button = gr.Button('Run')
        with gr.Row():
            mae_masked = gr.Image(label='Masked input', type='pil')
            mae_recon = gr.Image(label='Reconstruction', type='pil')
            mae_orig = gr.Image(label='Original', type='pil')
        mae_text = gr.Textbox(label='Land cover class')
        mae_button.click(run_mae, tile, [mae_masked, mae_recon, mae_orig, mae_text])

    with gr.Tab('VAE'):
        gr.Markdown('Variational autoencoder on Urdu characters.')
        vae_button = gr.Button('New random characters')
        vae_out = gr.Image(label='Samples from N(0, I)', type='pil')
        vae_button.click(vae_random, None, vae_out)
        with gr.Row():
            d1 = gr.Dropdown(class_names, value=class_names[0], label='First character')
            d2 = gr.Dropdown(class_names, value=class_names[20], label='Second character')
        alpha = gr.Slider(0, 1, value=0.5, step=0.05, label='Position between the two characters')
        inter_out = gr.Image(label='First character, decoded point, second character', type='pil')
        for c in [d1, d2, alpha]:
            c.change(interpolate, [d1, d2, alpha], inter_out)

    with gr.Tab('Diffusion'):
        gr.Markdown('Diffusion model. Pick a character and generate 8 samples. This takes a while.')
        char = gr.Dropdown(class_names, value=class_names[0], label='Character')
        diff_button = gr.Button('Generate 8 samples')
        diff_out = gr.Image(label='Generated samples', type='pil')
        diff_button.click(run_diffusion, char, diff_out)

    with gr.Tab('RAG'):
        gr.Markdown('Biomedical questions answered from PubMed abstracts.')
        question = gr.Textbox(label='Question', lines=2)
        rag_button = gr.Button('Answer')
        with gr.Row():
            answer = gr.Textbox(label='Answer', lines=8)
            notice = gr.Textbox(label='Notice', value=NOTICE)
        passages_box = gr.Textbox(label='Retrieved passages', lines=20)
        rag_button.click(run_rag, question, [answer, passages_box, notice])

demo.launch(share=os.path.exists('/content'))

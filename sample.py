import sys, time
import torch
import numpy as np
from transformers import GPT2LMHeadModel

np.set_printoptions(threshold=sys.maxsize)

t0 = time.time()
#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/music-gpt2-micro/checkpoint-399000")
#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/music-gpt2-mini3/checkpoint-400000").cuda()
#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/music-gpt2-mini-reg/checkpoint-400000").cuda()
#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/music-gpt2-small/checkpoint-200000").cuda()

#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/fma-gpt2-test/checkpoint-400000").cuda()
#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/fma-gpt2-small/checkpoint-400000").cuda()
#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/fma-gpt2-medium/checkpoint-160000").cuda()
#model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/fma-gpt2-medium/checkpoint-293000").cuda()

model = GPT2LMHeadModel.from_pretrained("/nlp/scr/jthickstun/fma-gpt2-small-bs512/checkpoint-35000").cuda()

print(f'Loaded model ({time.time()-t0} seconds)')
#print(model)

d = 1024

#input_ids = np.load('../mel2spec/wavs/rock-codes.npy')[:512]
#input_ids = np.load('../mel2spec/wavs/motown-codes.npy')[:512]
#input_ids = np.load('../mel2spec/wavs/piano-codes.npy')[:512]
#input_ids = np.load('../mel2spec/wavs/symphony-codes.npy')[:512]
#input_ids = np.load('../mel2spec/wavs/violin-codes.npy')[:512]
#input_ids = np.load('../mel2spec/wavs/vocals-codes.npy')[:512]
#input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

prompts = { 'unconditional' : None,
            'rock' : '../mel2spec/wavs/rock-codes.npy',
            'motown' : '../mel2spec/wavs/motown-codes.npy',
            'piano' : '../mel2spec/wavs/piano-codes.npy',
            'symphony' : '../mel2spec/wavs/symphony-codes.npy',
            'violin' : '../mel2spec/wavs/violin-codes.npy',
            'vocals' : '../mel2spec/wavs/vocals-codes.npy' }

for identifier, prompt in prompts.items():
    for i in range(5):
        t0 = time.time()
        if prompt is None:
            input_ids = torch.tensor([[65537]]).cuda()
            #input_ids = torch.tensor([[65536]]).cuda()
        else:
            input_ids = np.load(prompt)[:512]
            input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
        output_ids = []
        sample_output = model.generate(input_ids, do_sample=True, max_length=d, temperature=1.0)
        output_ids.append(sample_output[0])
        #for j in range(5):
        #    ids = sample_output[0][-d//2:].unsqueeze(0)
        #    sample_output = model.generate(ids, do_sample=True, max_length=d)
        #    output_ids.append(sample_output[0][:d//2])

        print("Output:\n" + 100 * "-")
        output_ids = torch.cat(output_ids)
        filename = f'codes/{identifier}-{i}.npy'
        print(filename, output_ids.shape)
        print(f'Sampling time: {time.time()-t0} seconds')
        if output_ids.shape[0] < 1024: continue
        np.save(filename, output_ids.cpu().numpy())

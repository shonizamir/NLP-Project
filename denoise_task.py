import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch
from omegaconf import OmegaConf
import clip
import torchaudio
from llama_inference.llama import Tokenizer, Llama
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.misc as misc
from codec.MSCodec import MSCodecLM
import random
import typing as tp
from collections import OrderedDict
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from pystoi.stoi import stoi

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

class DenoisingDataset(Dataset):
    def __init__(self, data_root, tsv_path, time, model_path, audio_tokenizer, text_token_embedding, device="cpu", induction=1,vq1_texts=None):
        self.device = device
        self.text_token_embedding = text_token_embedding
        self.data_root = data_root
        self.text_tokenizer = Tokenizer(model_path=model_path + "/tokenizer.model")
        self.induction = induction
        self.vq1_texts = list(vq1_texts)
        self.audio_tokenizer = audio_tokenizer

        self.noisy_paths = []
        self.clean_paths = []
        with open(tsv_path) as f:
            for line in f.readlines():
                image_ids = line.strip('\n').split(",")[:-1]
                curr_clean_paths = []
                curr_noisy_paths = []
                for image_id in image_ids:
                    clean_path, noisy_path = image_id.split("/")
                    curr_clean_paths.append(clean_path)
                    curr_noisy_paths.append(noisy_path)
                self.clean_paths.append(curr_clean_paths)
                self.noisy_paths.append(curr_noisy_paths)
        self.clean_paths = self.clean_paths[time:]
        self.noisy_paths = self.noisy_paths[time:]



    def __len__(self):
        return len(self.noisy_paths)

    def __getitem__(self, index):
        select_texts = []
        select_audios = []
        ###Instruction
        if self.induction == 0:
            instruction = ''
        else:
            instruction = "Please denoise the last noisy input"
            """instruction = (
            "You are an expert in speech enhancement. "
            "Given a noisy separated audio input from a previous speech separation step, "
            "your task is to generate a cleaner and more natural-sounding version of the same speaker's audio. "
            "The input is a quantized representation of audio. "
            "Please output a denoised version using the same quantized format."
            )"""

        prompt_tokens = torch.tensor(self.text_tokenizer.encode(instruction, bos=True, eos=False)).unsqueeze(0).to(self.device)
        in_tokens = torch.tensor(self.text_tokenizer.encode("###\ninput: < ", bos=False, eos=False), dtype=torch.int64).unsqueeze(0).to(self.device)
        out_tokens = torch.tensor(self.text_tokenizer.encode(" >\noutput: ", bos=False, eos=False), dtype=torch.int64).unsqueeze(0).to(self.device)
        prompt_features = self.text_token_embedding(prompt_tokens)
        # assert 1==2
        in_feature = self.text_token_embedding(in_tokens)
        out_feature = self.text_token_embedding(out_tokens)

        ###Load images and class texts
        noisy_ids = self.noisy_paths[index]
        clean_ids = self.clean_paths[index]
        last_setence = ''
        layer1_len = 0
        layer2_len = 0
        layer3_len = 0
        for i in range(0, len(noisy_ids)):
            noisy_id = noisy_ids[i]
    
            wav_root = os.path.join(self.data_root, noisy_id)
            wav, sr = torchaudio.load(wav_root)
            if sr != 16000:
                wav = convert_audio(wav, sr, 16000, 1)
            wav = wav.unsqueeze(1).to(self.device)
            if wav.shape[2]/16000 > 1:
                wav = wav[:,:,:1*16000]
            else:
                wav_new = torch.zeros(1, 1, 1*16000).type_as(wav)
                wav_new[:,:,:wav.shape[2]] = wav
                wav = wav_new
            my_code = []
            setence = ''
            with torch.no_grad():
                x, codes , _, _,_,_ = self.audio_tokenizer(wav)
                for kk, code in enumerate(codes):
               
                    #if kk != 0:
                        #continue
                    for j in range(code.shape[1]):
                        if kk==0:
                        
                            tmp = code[0,j].item() # index
                            wo = self.vq1_texts[tmp] # get word
                
                            real_code = self.text_tokenizer.encode(str(wo), bos=False, eos=False)
                            my_code += real_code
                            setence += ' ' + str(wo)
                            layer1_len = code.shape[1]
                        else:
                            if kk == 1 : 
                                layer2_len = code.shape[1]   
        
                            if kk == 2 :
                                layer3_len = code.shape[1]   
            
                            tmp = code[0,j].item()
                            wo = self.text_tokenizer.decode(tmp)
                            setence += ' ' + str(wo)
                            my_code.append(tmp)
                  
            if i == len(noisy_ids)-1:
                last_setence = setence
            #assert 1==2
            my_code = np.array(my_code)
            my_code = torch.from_numpy(my_code).to(self.device)
            select_audios.append(my_code)
            if i != len(noisy_ids)-1:
                select_audios.append(my_code)
     
            

        layers_len = [layer1_len, layer2_len, layer3_len]
        for i in range(0, len(clean_ids)):
            clean_id = clean_ids[i]
            wav_root = os.path.join(self.data_root, clean_id)
            wav, sr = torchaudio.load(wav_root)
            if sr != 16000:
                wav = convert_audio(wav, sr, 16000, 1)
            wav = wav.unsqueeze(1).to(self.device)
            if wav.shape[2]/16000 > 1:
                wav = wav[:,:,:1*16000]
            else:
                wav_new = torch.zeros(1, 1, 1*16000).type_as(wav)
                wav_new[:,:,:wav.shape[2]] = wav
                wav = wav_new
            my_code = []
            setence = ''
            with torch.no_grad():
                x, codes , _, _,_,_ = self.audio_tokenizer(wav)
                for kk, code in enumerate(codes):
                    #if kk != 0:
                        #continue
                    for j in range(code.shape[1]):
                        if kk==0:
                            tmp = code[0,j].item() # index
                            wo = self.vq1_texts[tmp] # get word
                            real_code = self.text_tokenizer.encode(str(wo), bos=False, eos=False)
                            my_code += real_code
                            setence += ' ' + str(wo)
                        else:
                            tmp = code[0,j].item()
                            wo = self.text_tokenizer.decode(tmp)
                            setence += ' ' + str(wo)
                            my_code.append(tmp)
        
            my_code = np.array(my_code)
            my_code = torch.from_numpy(my_code).to(self.device)
            select_texts.append(my_code)
            if i != len(clean_ids)-1:
                select_texts.append(my_code)
            
            
        ##The last image serves query image (GT)
        target_texts = select_texts[-1] # the last one as the target
        select_texts = select_texts[:-1] # previous

        ##Generating context examples with other images
        for i in range(0, len(select_texts)):
            text_token = select_texts[i].unsqueeze(0)
            text_feature = self.text_token_embedding(text_token)
            vis_token = select_audios[i].unsqueeze(0)
            vis_feature = self.text_token_embedding(vis_token)
            prompt_tokens = torch.cat([prompt_tokens, in_tokens, vis_token, out_tokens, text_token], dim=-1)
            prompt_features = torch.cat( [prompt_features, in_feature, vis_feature , out_feature, text_feature], dim=1)

        ##Adding query token
        vis_texts = ""
        vis_token = select_audios[-1].unsqueeze(0)
        prompt_tokens = torch.cat([prompt_tokens, in_tokens, vis_token, out_tokens], dim=-1)
        prompt_features = torch.cat( [prompt_features, in_feature, self.text_token_embedding(vis_token), out_feature], dim=1)
        
        prompt_tokens = prompt_tokens[0].to("cpu")
        prompt_features = prompt_features[0].to("cpu")
        return [prompt_tokens, prompt_features, target_texts, clean_ids, layers_len]

 
def custom_collate_fn(batch):
    prompt_tokens = torch.stack([item[0] for item in batch])
    prompt_features = torch.stack([item[1] for item in batch])
    target_codes = [torch.stack([item[2][i] for item in batch]) for i in range(3)]
    noisy_paths = [item[3] for item in batch]
    clean_paths = [item[4] for item in batch]
    return [prompt_tokens, prompt_features, target_codes, noisy_paths, clean_paths]

def get_args_parser():
    parser = argparse.ArgumentParser("Audio Denoising", add_help=False)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--llama_model_path", default="./llama", type=str)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--output_dir", default="./output_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--audio_path", default="/data/all", type=str)
    parser.add_argument("--file_path", default="/data/all", type=str)
    parser.add_argument("--vqgan_path", default="vqgan_weight/vqgan_imagenet_f16_16384", type=str)
    parser.add_argument("--n_vision_words", default=32000, type=int)
    parser.add_argument("--output_type", default="next_token_prediction", type=str)
    parser.add_argument("--vq_config_path", type=str, default="vqgan_configs/model_16384.yaml")
    parser.add_argument("--codec_ckpt", type=str, default="stage_1_llama_fix-40.pth")
    parser.add_argument("--induction", type=int, default=2)
    return parser

def si_sdr(estimated, original, eps=1e-8):
    estimated = estimated.squeeze()
    original = original.squeeze()
    assert estimated.shape == original.shape, f"Shape mismatch: {estimated.shape} vs {original.shape}"
    dot = torch.sum(estimated * original)
    s_target = dot / (torch.sum(original ** 2) + eps) * original
    e_mix = estimated - s_target
    s_target_energy = torch.sum(s_target ** 2) + eps
    e_mix_energy = torch.sum(e_mix ** 2) + eps
    si_sdr_value = 10 * torch.log10(s_target_energy / e_mix_energy)
    return si_sdr_value.item()

    
def main(args):
    misc.init_distributed_mode(args)
   

    device = torch.device(args.device)
    torch.manual_seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    cudnn.benchmark = True
    vq1_texts = np.load("./layer1.npy", allow_pickle=True)

    exp_model_config = OmegaConf.load(args.vq_config_path)
    model = MSCodecLM(**exp_model_config.generator.config)
    parameter_dict = torch.load(args.codec_ckpt, weights_only=False)
    model.load_state_dict(parameter_dict['codec_model'])
    model.to(device)
    model.eval()


    generator = Llama.build(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.llama_model_path + "/tokenizer.model",
        max_seq_len=args.max_seq_len,
        max_batch_size=2,
    )
    llama_vocab_size = generator.tokenizer.n_words
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()


    os.makedirs(args.output_dir, exist_ok=True)
    dataset = DenoisingDataset(
        data_root=args.audio_path, tsv_path=args.file_path, time=0,
        model_path=args.llama_model_path, audio_tokenizer=model,
        text_token_embedding=generator.model.tok_embeddings, device=device,
        induction=args.induction, vq1_texts=vq1_texts
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=False, drop_last=False
    )

    total_si_sdr = 0
    total_samples = 0
    total_stoi = 0
    total_snr = 0
    metric_logger = misc.MetricLogger(delimiter="  ")
    i = 0
    max_sdr_value = float('-inf')
    max_sdr_reconstructed = ""
    max_sdr_clean = ""
  
    
    for data_iter_step, [prompt_tokens, prompt_features, target_texts, clean_ids, layers_len] in enumerate(
        metric_logger.log_every(data_loader, 10, "Denoising")
    ):
        
        prompt_tokens = prompt_tokens.to(device)
        prompt_features = prompt_features.to(device)

       
        out_mask = torch.ones(args.batch_size, args.max_seq_len, llama_vocab_size).to(device)
        predictions = generator.generate_fewshot(
            prompt_tokens,
            prompt_features,
            induction=args.induction,
            out_mask=out_mask,
            max_gen_len=100,
            temperature=0,
            top_p=1.0,
        )
        clean_p = clean_ids[-1][0]
        for idx, (pred, target_text) in enumerate(zip(predictions, target_texts)):
            pred_tokens = pred['tokens']  # Extract predicted tokens
            # Step 2: Decode LLaMA tokens using LLaMA's text tokenizer to get words
            decoded_words = [generator.tokenizer.decode(t) for t in pred_tokens]
            stop_token = "###"
            # Find the index of stop token
            if stop_token in decoded_words:
                stop_index = decoded_words.index(stop_token)
                decoded_words = decoded_words[:stop_index]
            # Step 3: Map each word back to its corresponding VQ token
            layers = [t.item() for t in layers_len]
            try : 
                vq1_indices = [dataset.vq1_texts.index(word) for word in decoded_words[:layers[0]]]  # Map words to VQ indices
            except ValueError:
                print("couldnt find word in vq1_texts. continuing")
                continue
            vq1 = torch.tensor(vq1_indices).unsqueeze(0).to(device)
            vq2 = torch.tensor(pred_tokens[layers[0]:layers[0]+layers[1]]).unsqueeze(0).to(device)
            vq3 = torch.tensor(pred_tokens[layers[0]+layers[1]:layers[0]+layers[1]+layers[2]]).unsqueeze(0).to(device)
            # Step 4: Convert the list of VQ indices into a tensor and move it to the device
              # Add batch dimension
            # Step 5: Decode the VQ tokens back into the waveform using the audio tokenizer
            tensor_list = [vq1, vq2, vq3]
            with torch.no_grad():
                reconstructed_wav, *_ = model.decode(tensor_list)
                # Save the output wav file
                decoded_audio = reconstructed_wav.squeeze(1)
                if len(clean_p.split('s1-')) == 2:
                    number_part = clean_p.split('s1-')[1]
                else:
                    number_part = clean_p.split('s2-')[1]
                file_name = f'recontructed-{number_part}.wav'
                output_path = os.path.join(args.output_dir, file_name)
                try:
                    torchaudio.save(output_path, decoded_audio.cpu(), sample_rate=16000)
                    print(f"Saved Separated audio to: {output_path}")
                except Exception as e:
                    print(f"Failed to save Separated audio: {e}")
                i+=1
            # Compute SI-SDR
            clean_wav_root = os.path.join(dataset.data_root, clean_p)
            clean_wav, clean_sr = torchaudio.load(clean_wav_root)
            if clean_sr != 16000:
                clean_wav = convert_audio(clean_wav, clean_sr, 16000, 1)
            clean_length = clean_wav.shape[1]
            clean_wav = clean_wav.unsqueeze(1).to(device)
            min_length = min(decoded_audio.shape[1], clean_wav.shape[1])
            decoded_audio = decoded_audio[:, :min_length]
            clean_wav_eval = clean_wav.squeeze(1)[:, :min_length]
            
            si_sdr_value = si_sdr(decoded_audio, clean_wav_eval)
            
            if(si_sdr_value > max_sdr_value):
                max_sdr_value = si_sdr_value
                max_sdr_reconstructed = output_path
                max_sdr_clean = clean_wav_root
                
            
            
            clean_wav_np = clean_wav_eval.cpu().numpy()  # Move to CPU and convert to NumPy
            decoded_audio_np = decoded_audio.cpu().numpy()  # Move to CPU and convert to NumPy
            total_si_sdr += si_sdr_value
            total_samples += 1

            print(f"Clean_p: {clean_p}, SI-SDR: {si_sdr_value:.4f} dB, Saved to: {output_path}")

    avg_si_sdr = total_si_sdr / total_samples if total_samples > 0 else 0

    print(f"Average SI-SDR: {avg_si_sdr:.4f} dB\n")
    print(f"MAX SDR value: {max_sdr_value:.4f} dB, for the reconstructed file: {max_sdr_reconstructed}\n and clean: {max_sdr_clean}\n")
    

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.batch_size = 1
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
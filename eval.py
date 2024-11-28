import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio
from df import enhance, init_df
from os.path import join as pjoin
import matplotlib.pyplot as plt
import os
import tqdm
import time
from IPython.display import Audio #listen: ipd.Audio(real.detach().cpu().numpy(), rate=FS)
import numpy as np
import scipy.signal as sig
import pandas as pd
import torchmetrics.audio as M
from speechmos import dnsmos
from datetime import datetime

def rep_list(short, long):
    reps = int(np.ceil(len(long) / len(short)))
    short *= reps
    short = short[:len(long)]
    return short
    
def plot_tensor(x):
    plt.plot(x.cpu().detach().numpy())



def extend_signal(signal, target_length):
    """
    Extend a signal by repeating it if it's shorter than the target length.
    
    Args:
    signal (torch.Tensor): Input signal.
    target_length (int): Desired length of the extended signal.

    Returns:
    torch.Tensor: Extended signal.
    """
    current_length = signal.size(0)
    if current_length < target_length:
        repetitions = target_length // current_length
        remainder = target_length % current_length
        extended_signal = signal.repeat(repetitions)
        if remainder > 0:
            extended_signal = torch.cat((extended_signal, signal[:remainder]), dim=0)
        return extended_signal
    else:
        return signal

def load_audio(apath):
    audio, fs = torchaudio.load(apath)
    if fs != FS:
        #print('resampling')
        resampler = torchaudio.transforms.Resample(fs, FS)
        audio = resampler(audio)    
    if len(audio.shape) > 1:
            audio = audio[0,:]
    return audio

def power(signal):
    return np.mean(signal**2)
    
def add_batch_results(meta, pesq, stoi, sisdri, srmr, dnsmos_result, model_name, batch_size, df):
    idx = len(df)
    for i in range(batch_size):
        df.loc[idx+i, 'train_rirs'] = TRAINRIR_NAMES[model_name]
        df.loc[idx+i, 'eval_rirs'] = rir_path.split('/')[-1].split('_')[0]
        df.loc[idx+i, 'model'] = model_name
        df.loc[idx+i, 'speech'] = os.path.join(*meta[0][i].split('/')[6:])
        df.loc[idx+i, 'noise'] = os.path.join(*meta[1][i].split('/')[6:])
        df.loc[idx+i, 'rir'] = os.path.join(*meta[2][i].split('/')[4:])
        df.loc[idx+i, 'noisy_snr'] = meta[3][i].item()
        df.loc[idx+i, 'sisdri'] = sisdri[i]
        df.loc[idx+i, 'pesq'] = pesq[i]
        df.loc[idx+i, 'stoi'] = stoi[i]
        df.loc[idx+i, 'srmr'] = srmr[i]        
        df.loc[idx+i, 'ovrl_mos'] = dnsmos_result[i]['ovrl_mos']
        df.loc[idx+i, 'sig_mos'] = dnsmos_result[i]['sig_mos']
        df.loc[idx+i, 'bak_mos'] = dnsmos_result[i]['bak_mos']
        df.loc[idx+i, 'p808_mos'] = dnsmos_result[i]['p808_mos']
    return df
    
class DFN_dataset(Dataset):
    def __init__(self, speech_path, noise_path, rir_path, reverberant_noises):
        print('Initializing dataset...')
        self.speech_path = speech_path
        self.noise_path = noise_path
        self.rir_path = rir_path
        self.reverberant_noises = reverberant_noises
        
        # load speech paths
        self.speech_paths = []
        with open(speech_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.speech_paths.append(line.rstrip()) 
        errors = []
        with open('dns_test_errors.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                errors.append(line.rstrip()) 
        self.speech_paths = [item for item in self.speech_paths if item not in errors]

        self.snrs = np.random.uniform(low = 0, high = 30, size = len(self.speech_paths))
        # load noise paths
        self.noise_paths = []
        with open(noise_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.noise_paths.append(line.rstrip()) 

        # load rir paths
        self.rir_paths = []
        with open(rir_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.rir_paths.append(line.rstrip()) 

        self.noise_paths = rep_list(self.noise_paths, self.speech_paths)
        self.rir_paths = rep_list(self.rir_paths, self.speech_paths)
        print('All paths loaded.')
        
    def __len__(self):
        return len(self.speech_paths)


    def __getitem__(self, idx):
        clean = load_audio(self.speech_paths[idx])
        # handle weird case where speech is silence
        if len(clean) >= FS*DURATION:
            speech_nrgy = torch.mean(clean[:FS*DURATION]**2)
        else:
            speech_nrgy = torch.mean(clean **2)
        if speech_nrgy == 0:
            clean = load_audio(self.speech_paths[0])

        noise = load_audio(self.noise_paths[int(idx % len(self.noise_paths))])

        # handle silent rir
        rir = load_audio(self.rir_paths[int(idx % len(self.rir_paths))])
        rir_nrgy = torch.mean(rir**2)
        if rir_nrgy == 0:
            #print('silent rir')
            rir = torch.zeros(FS)
            rir[300] = 1.


        # we extend speech and noise if too short
        if len(clean) < FS * DURATION:
            clean = extend_signal(clean, FS*DURATION)
        if len(noise) < FS * DURATION:
            noise = extend_signal(noise, FS*DURATION)

        # back to numpy for easy conv
        clean = clean.numpy()
        noise = noise.numpy()
        rir = rir.numpy()
            
        # we choose the signal chunk with more energy (to avoid silent chunks)
        nchunks = len(clean) // (FS*DURATION)
        chunks = np.split(clean[: FS * DURATION * nchunks], nchunks)
        powers = np.array([power(x) for x in chunks])
        clean = clean[np.argmax(powers) * FS * DURATION : (np.argmax(powers) + 1 ) *  FS * DURATION]
        
        nchunks = len(noise) // (FS*DURATION)
        chunks = np.split(noise[: FS * DURATION * nchunks], nchunks)
        powers = np.array([power(x) for x in chunks])
        noise = noise[np.argmax(powers) * FS * DURATION : (np.argmax(powers) + 1 ) *  FS * DURATION]

        #handle silent noise
        noise_nrgy = power(noise)
        if noise_nrgy == 0.:
            #print('silent noise sample, using white noise')
            noise = np.random.randn( FS * DURATION )

        # we set the SNR
        ini_snr = 10 * np.log10(power(clean) / power(noise))
        noise_gain_db = ini_snr - self.snrs[idx]
        noise *= np.power(10, noise_gain_db/20)

        # we normalize to 0.9 if mixture is close to clipping
        clips = np.max(np.abs(clean + noise))
        if clips >= 0.9:
            clips /= 0.9
            noise /= clips
            clean /= clips
        # or to -18dBfs if smaller than that:
        elif clips <= 10**(-18/20):
            clips /= 10**(-18/20)
            noise /= clips 
            clean /= clips    

        # apply rir 
        revspeech = sig.fftconvolve(clean, rir, 'full')
        # synchronize reverberant with anechoic
        lag = np.where(np.abs(rir) >= 0.5*np.max(np.abs(rir)))[0][0]

        revspeech = revspeech[lag:FS*DURATION + lag]

        # enforce energy conservation
        revspeech *= np.sqrt(power(clean) / power(revspeech)) 

        # apply RIR to noise too if needed
        if self.reverberant_noises:
            rnoise = sig.fftconvolve(noise, rir, 'full')
            rnoise = rnoise[lag:FS*DURATION + lag]
            rnoise *= np.sqrt(power(noise) / power(rnoise))
            noise = rnoise
        noisy = revspeech + noise
        
        # check for Nans
        if np.any(np.isnan(noisy)):
            print('noisy nan')
        if np.any(np.isnan(clean)):
            print('clean nan')
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)
        meta = [self.speech_paths[idx], self.noise_paths[int(idx % len(self.noise_paths))], self.rir_paths[int(idx % len(self.rir_paths))], self.snrs[idx].item()]
        return noisy.float(), clean.float(), meta
# dataset check
'''
for rir_path in rir_paths:
    print(rir_path)
    dataset = DFN_dataset(speech_path, noise_path, rir_path, reverberant_noises)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True) 
    for x in tqdm.tqdm(dataloader):
        noisy, clean = x
''';
if __name__ == '__main__':
    FS = 48000
    DURATION = 4 #time in seconds of the eval chunk
    TRAINRIR_NAMES = {'D01_sb_none_NH_mono': 'singleband' , 'D02_mb_none_NH_mono': 'multiband', 
                'D03_mb_rec_NH_left': 'recdirectivity', 'D05_mb_srcrec_NH_left': 'recsourcedirectivity',
                'D00_DNS5': 'DNS5'}

    use_gpu = True
    if torch.cuda.is_available() and use_gpu:
        TORCH_DEVICE = "cuda"
    else:
        TORCH_DEVICE = "cpu"

    batch_size = 1
    num_workers = 8
    reverberant_noises = True
    speech_path = '/home/ubuntu/Data/DFN/textfiles/test_set.txt'
    noise_path = '/home/ubuntu/Data/DFN/textfiles/test_set_noise.txt'
    dns_mos_path = '/home/ubuntu/enric/DNS-Challenge/DNSMOS/DNSMOS'

    rir_paths = ['/home/ubuntu/enric/guso_interspeech24/real_rirs.txt',
    '/home/ubuntu/Data/DFN/textfiles/singleband_test_rir.txt',
    '/home/ubuntu/Data/DFN/textfiles/multiband_test_rir.txt',
    '/home/ubuntu/Data/DFN/textfiles/recdirectivity_left_test_rir.txt',
    '/home/ubuntu/Data/DFN/textfiles/recsourcedirectivity_left_test_rir.txt']

    model_names = list(TRAINRIR_NAMES.keys())
    results_path = 'results'

    np.random.seed(0)
    torch.manual_seed(0)


    # Disable the GPU
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]=""
    if not os.path.exists(results_path):
        # Create the directory
        os.makedirs(results_path)
        
    downsampler = torchaudio.transforms.Resample(FS, 16000)
    pesq_single = M.PerceptualEvaluationSpeechQuality(16000, 'wb').to(TORCH_DEVICE)
    stoi_single = M.ShortTimeObjectiveIntelligibility(FS, True).to(TORCH_DEVICE)
    sisdr_single = M.ScaleInvariantSignalDistortionRatio().to(TORCH_DEVICE)
    srmr_single = M.SpeechReverberationModulationEnergyRatio(FS).to(TORCH_DEVICE)

    for rir_path in rir_paths: #for each set of eval RIRs
        for model_name in model_names: #for each model (or set of training RIRs)
            df = pd.DataFrame(columns=['train_rirs', 'eval_rirs', 'model', 'speech', 'noise', 'rir', 'noisy_snr', 'sisdri',
                                'pesq', 'stoi', 'srmr', 'ovrl_mos', 'sig_mos', 'bak_mos', 'p808_mos'])
            model_path = pjoin('/home/ubuntu/Data/DFN', model_name)
            dataset = DFN_dataset(speech_path, noise_path, rir_path, reverberant_noises)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True) 
            model, df_state, _ = init_df(model_path)
            
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Running evaluation of '+model_name+' evaluated on '+
                rir_path.split('/')[-1].split('_')[0]+'RIRs ...')
            for noisy, clean, meta in tqdm.tqdm(dataloader):
                try:
                    enhanced = enhance(model, df_state, noisy)
                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error while enhancing '+os.path.join(*meta[0][0].split('/')[6:]))

                downsampler = downsampler.to('cpu')
                ds_enhanced = downsampler(enhanced)
                didx = len(df)
                for i in range(batch_size):
                    df.loc[didx+i, 'train_rirs'] = TRAINRIR_NAMES[model_name]
                    df.loc[didx+i, 'eval_rirs'] = rir_path.split('/')[-1].split('_')[0]
                    df.loc[didx+i, 'model'] = model_name
                    df.loc[didx+i, 'speech'] = os.path.join(*meta[0][i].split('/')[6:])
                    df.loc[didx+i, 'noise'] = os.path.join(*meta[1][i].split('/')[6:])
                    df.loc[didx+i, 'rir'] = os.path.join(*meta[2][i].split('/')[4:])
                    df.loc[didx+i, 'noisy_snr'] = meta[3][i].item()
                    try:
                        # first the CPU metrics
                        dnsmos_result = dnsmos.run(ds_enhanced[i].numpy(), 16000) 
                        df.loc[didx+i, 'ovrl_mos'] = dnsmos_result['ovrl_mos']
                        df.loc[didx+i, 'sig_mos'] = dnsmos_result['sig_mos']
                        df.loc[didx+i, 'bak_mos'] = dnsmos_result['bak_mos']
                        df.loc[didx+i, 'p808_mos'] = dnsmos_result['p808_mos']
                    except:
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting DNSMOS of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                        df.loc[didx+i, 'ovrl_mos'] = np.NAN
                        df.loc[didx+i, 'sig_mos'] = np.NAN
                        df.loc[didx+i, 'bak_mos'] = np.NAN
                        df.loc[didx+i, 'p808_mos'] = np.NAN   
                # then to GPU
                ds_enhanced.to(TORCH_DEVICE)
                noisy = noisy.to(TORCH_DEVICE)
                clean = clean.to(TORCH_DEVICE)
                enhanced = enhanced.to(TORCH_DEVICE)
                downsampler_gpu = downsampler.to(TORCH_DEVICE)
                for i in range(batch_size):
                    try:
                        sisdr_final = sisdr_single(enhanced[i], clean[i]).item() 
                        sisdr_original = sisdr_single(noisy[i], clean[i]).item() 
                        sisdri = sisdr_final - sisdr_original
                    except:
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SISDR of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)                    
                        sisdri = np.NAN
                    df.loc[didx+i, 'sisdri'] = sisdri
                    try:
                        df.loc[didx+i, 'pesq'] = pesq_single(ds_enhanced[i], downsampler_gpu(clean[i])).item()
                    except:
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting PESQ of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                        df.loc[didx+i, 'pesq'] = np.NAN
                    try:
                        df.loc[didx+i, 'stoi'] = stoi_single(enhanced[i], clean[i]).item()
                    except:
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting STOI of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)                    
                        df.loc[didx+i, 'stoi'] = np.NAN
                    try:
                        df.loc[didx+i, 'srmr'] = srmr_single(enhanced[i]).item()
                    except:
                        df.loc[didx+i, 'srmr'] = np.NAN
                        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SRMR of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
            df.to_csv(pjoin(results_path, model_name+'_evaluatedOn_'+rir_path.split('/')[-1].split('_')[0]), index=False)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Done.')

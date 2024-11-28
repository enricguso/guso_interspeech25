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
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
import evaluation_metrics.calculate_intrusive_se_metrics as intru
import evaluation_metrics.NISQA.nisqa.NISQA_lib as NL
from evaluation_metrics.nisqa_utils import load_nisqa_model
from evaluation_metrics.nisqa_utils import predict_nisqa

import evaluation_metrics.calculate_phoneme_similarity as phon
from evaluation_metrics.calculate_phoneme_similarity import LevenshteinPhonemeSimilarity

from espnet2.bin.spk_inference import Speech2Embedding
import evaluation_metrics.calculate_speaker_similarity as spksim

from discrete_speech_metrics import SpeechBERTScore
import evaluation_metrics.calculate_speechbert_score as sbert
import evaluation_metrics.calculate_wer as wer

#HELPER FUCTIONS
rir_path = '_'
def rep_list(short, long):
    #repeat a list until the length of a longer one
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
    # audio loader in torch
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
    
# test set dataset class, dataloader. on-the-fly mixture generation
class DFN_dataset(Dataset):
    def __init__(self, speech_path):
        # we store the textfile path in the class
        print('Initializing dataset...')
        self.speech_path = speech_path

        # load speech wav paths from the textfile
        self.speech_paths = []
        with open(speech_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.speech_paths.append(line.rstrip()) 
        print('All paths loaded.')
        
    def __len__(self):
        return len(self.speech_paths)


    def __getitem__(self, idx):
        # generate the mix
        clean = load_audio(self.speech_paths[idx])
        # handle weird case where speech is silence
        if len(clean) >= FS*DURATION:
            speech_nrgy = torch.mean(clean[:FS*DURATION]**2)
        else:
            speech_nrgy = torch.mean(clean **2)
        if speech_nrgy == 0:
            clean = load_audio(self.speech_paths[0])


        # we extend speech and noise if too short
        if len(clean) < FS * DURATION:
            clean = extend_signal(clean.numpy(), FS*DURATION)

        '''    
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
        '''
        # we normalize to 0.9 if mixture is close to clipping
        clips = torch.max(torch.abs(clean))
        if clips >= 0.9:
            clips /= 0.9
            clean /= clips
        # or to -18dBfs if smaller than that:
        elif clips <= 10**(-18/20):
            clips /= 10**(-18/20)
            clean /= clips    


        # check for Nans

        if torch.any(torch.isnan(clean)):
            print('clean nan')
        #clean = torch.from_numpy(clean)
        meta = [self.speech_paths[idx]]
        return clean.float(), meta

'''
# check data integrity
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

    # model name and alias
    TRAINRIR_NAMES = {'D01_sb_none_NH_mono': 'singleband' , 'D02_mb_none_NH_mono': 'multiband', 
                'D03_mb_rec_NH_left': 'recdirectivity', 'D05_mb_srcrec_NH_left': 'recsourcedirectivity',
                'D00_DNS5': 'DNS5', 'D09_SSmp3d_left' : 'soundspaces'}

    use_gpu = True
    if torch.cuda.is_available() and use_gpu:
        TORCH_DEVICE = "cuda"
    else:
        TORCH_DEVICE = "cpu"

    batch_size = 1
    num_workers = 8
    reverberant_noises = True
    speech_path = '/home/ubuntu/Data/DFN/textfiles/DNS5_val_speakerphone.txt'
    dns_mos_path = '/home/ubuntu/enric/DNS-Challenge/DNSMOS/DNSMOS' #needed for the DNSMOS metrics

    #'/home/ubuntu/Data/DFN/textfiles/multiband_test_rir.txt',
    #'/home/ubuntu/Data/DFN/textfiles/singleband_test_rir.txt',
    #'/home/ubuntu/Data/DFN/textfiles/recdirectivity_left_test_rir.txt',
    #'/home/ubuntu/Data/DFN/textfiles/recsourcedirectivity_left_test_rir.txt']
    model_names = list(TRAINRIR_NAMES.keys())
    results_path = 'results_speakerphone'

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
    #srmr_single = M.SpeechReverberationModulationEnergyRatio(FS).to(TORCH_DEVICE)

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    objective_model = SQUIM_OBJECTIVE.get_model()

    subjective_model = subjective_model.to(TORCH_DEVICE)
    objective_model = objective_model.to(TORCH_DEVICE)

    NMR_SPEECH = torchaudio.utils.download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")
    nmr_speech, _ = torchaudio.load(NMR_SPEECH)
    nmr_speech = nmr_speech.to(TORCH_DEVICE)

    nisqa_model = load_nisqa_model('evaluation_metrics/NISQA/weights/nisqa.tar', 'cuda')
    phon_model = LevenshteinPhonemeSimilarity(device='cuda')
    phon_model.phoneme_predictor.eval();
    spksim_model = Speech2Embedding.from_pretrained(
        model_tag="espnet/voxcelebs12_rawnet3", device='cuda')
    spksim_model.spk_model.eval();
    sbs_model = SpeechBERTScore()
    sbs_model.model.eval();

    for model_name in model_names: #for each model (or set of training RIRs)

        #initialize the csv file
        df = pd.DataFrame(columns=['train_rirs', 'eval_rirs', 'model', 'speech', 'noise', 'rir', 'noisy_snr', 
                            
                            'dnsmos_ovrl_e', 'dnsmos_sig_e', 'dnsmos_bak_e', 'dnsmos_p808_e',
                            'squim_mos_e', 'squim_stoi_e', 'squim_pesq_e', 'squim_sisdr_e',
                            'nisqa_mos_e', 'nisqa_noi_e', 'nisqa_dis_e', 'nisqa_col_e',
                            'nisqa_loud_e',

                            #'sisdr_c', 'pesq_c', 'stoi_c', 'srmr_c', 
                            #'estoi_c', 'mcd_c', 'pesq2_c', 'sdr_c', 

                            'stoi_n', #'srmr_n', 
                            'dnsmos_ovrl_n', 'dnsmos_sig_n', 'dnsmos_bak_n', 'dnsmos_p808_n',
                            'squim_mos_n', 'squim_stoi_n', 'squim_pesq_n', 'squim_sisdr_n',
                            'nisqa_mos_n', 'nisqa_noi_n', 'nisqa_dis_n', 'nisqa_col_n',
                            'nisqa_loud_n', 

                            'dnsmos_ovrl_i', 'dnsmos_sig_i', 'dnsmos_bak_i', 'dnsmos_p808_i',
                            'squim_mos_i', 'squim_stoi_i', 'squim_pesq_i', 'squim_sisdr_i',
                            'nisqa_mos_i', 'nisqa_noi_i', 'nisqa_dis_i', 'nisqa_col_i',
                            'nisqa_loud_i'
                            ])


        #declare model and dataset  
        model_path = pjoin('/home/ubuntu/Data/DFN', model_name)
        dataset = DFN_dataset(speech_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True) 
        model, df_state, _ = init_df(model_path)
        
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Running evaluation of '+model_name+' evaluated on Speakerphone.')
        for noisy, meta in tqdm.tqdm(dataloader):

            try:
                enhanced = enhance(model, df_state, noisy)
            except:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error while enhancing '+os.path.join(*meta[0][0].split('/')[6:]))

            downsampler = downsampler.to('cpu')
            ds_enhanced = downsampler(enhanced)
            ds_noisy = downsampler(noisy)

            didx = len(df)
            for i in range(batch_size):
                df.loc[didx+i, 'train_rirs'] = TRAINRIR_NAMES[model_name]
                df.loc[didx+i, 'eval_rirs'] = 'none_speakerphone'
                df.loc[didx+i, 'model'] = model_name
                df.loc[didx+i, 'speech'] = os.path.join(*meta[0][i].split('/')[6:])                
                try:
                # first the DNSMOS metrics
                    dnsmos_result_e = dnsmos.run(ds_enhanced[i].numpy(), 16000) 
                    dnsmos_result_n = dnsmos.run(ds_noisy[i].numpy(), 16000) 

                    df.loc[didx+i, 'dnsmos_ovrl_e'] = dnsmos_result_e['ovrl_mos']
                    df.loc[didx+i, 'dnsmos_sig_e'] = dnsmos_result_e['sig_mos']
                    df.loc[didx+i, 'dnsmos_bak_e'] = dnsmos_result_e['bak_mos']
                    df.loc[didx+i, 'dnsmos_p808_e'] = dnsmos_result_e['p808_mos']

                    df.loc[didx+i, 'dnsmos_ovrl_n'] = dnsmos_result_n['ovrl_mos']
                    df.loc[didx+i, 'dnsmos_sig_n'] = dnsmos_result_n['sig_mos']
                    df.loc[didx+i, 'dnsmos_bak_n'] = dnsmos_result_n['bak_mos']
                    df.loc[didx+i, 'dnsmos_p808_n'] = dnsmos_result_n['p808_mos']

                    df.loc[didx+i, 'dnsmos_ovrl_i'] = dnsmos_result_e['ovrl_mos'] - dnsmos_result_n['ovrl_mos']
                    df.loc[didx+i, 'dnsmos_sig_i'] = dnsmos_result_e['sig_mos'] - dnsmos_result_n['sig_mos']
                    df.loc[didx+i, 'dnsmos_bak_i'] = dnsmos_result_e['bak_mos'] - dnsmos_result_n['bak_mos']
                    df.loc[didx+i, 'dnsmos_p808_i'] = dnsmos_result_e['p808_mos'] - dnsmos_result_n['p808_mos']
                    
                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting DNSMOS of '+os.path.join(*meta[0][i].split('/')[6:])+'__'+model_name)

                    df.loc[didx+i, 'dnsmos_ovrl_e'] = np.NAN
                    df.loc[didx+i, 'dnsmos_sig_e'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_bak_e'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_p808_e'] =  np.NAN

                    df.loc[didx+i, 'dnsmos_ovrl_n'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_sig_n'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_bak_n'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_p808_n'] =  np.NAN

                    df.loc[didx+i, 'dnsmos_ovrl_i'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_sig_i'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_bak_i'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_p808_i'] =  np.NAN
                
                try:
                    torchaudio.save('tmp.wav', enhanced, FS)
                    nisqa_sco = predict_nisqa(nisqa_model, 'tmp.wav')
                    df.loc[didx+i, 'nisqa_mos_e'] = nisqa_sco['mos_pred']
                    df.loc[didx+i, 'nisqa_noi_e'] = nisqa_sco['noi_pred']
                    df.loc[didx+i, 'nisqa_dis_e'] = nisqa_sco['dis_pred']
                    df.loc[didx+i, 'nisqa_col_e'] = nisqa_sco['col_pred']
                    df.loc[didx+i, 'nisqa_loud_e'] = nisqa_sco['loud_pred']

                    torchaudio.save('tmp.wav', noisy, FS)
                    nisqa_sco = predict_nisqa(nisqa_model, 'tmp.wav')
                    df.loc[didx+i, 'nisqa_mos_n'] = nisqa_sco['mos_pred']
                    df.loc[didx+i, 'nisqa_noi_n'] = nisqa_sco['noi_pred']
                    df.loc[didx+i, 'nisqa_dis_n'] = nisqa_sco['dis_pred']
                    df.loc[didx+i, 'nisqa_col_n'] = nisqa_sco['col_pred']
                    df.loc[didx+i, 'nisqa_loud_n'] = nisqa_sco['loud_pred']

                    df.loc[didx+i, 'nisqa_mos_i'] = df.loc[didx+i, 'nisqa_mos_e'] - df.loc[didx+i, 'nisqa_mos_n']
                    df.loc[didx+i, 'nisqa_noi_i'] = df.loc[didx+i, 'nisqa_noi_e'] - df.loc[didx+i, 'nisqa_noi_n']
                    df.loc[didx+i, 'nisqa_dis_i'] = df.loc[didx+i, 'nisqa_dis_e'] - df.loc[didx+i, 'nisqa_dis_n']
                    df.loc[didx+i, 'nisqa_col_i'] = df.loc[didx+i, 'nisqa_col_e'] - df.loc[didx+i, 'nisqa_col_n']
                    df.loc[didx+i, 'nisqa_loud_i'] = df.loc[didx+i, 'nisqa_loud_e'] - df.loc[didx+i, 'nisqa_loud_n']
                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting NISQA of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                

            # then to GPU
            ds_enhanced = ds_enhanced.to(TORCH_DEVICE)
            ds_noisy = ds_noisy.to(TORCH_DEVICE)
            noisy = noisy.to(TORCH_DEVICE)
            enhanced = enhanced.to(TORCH_DEVICE)
            downsampler_gpu = downsampler.to(TORCH_DEVICE)
            for i in range(batch_size):
                try:
                    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(ds_enhanced)
                    mos = subjective_model(ds_enhanced, nmr_speech)
                    df.loc[didx+i, 'squim_mos_e'] = mos.item()
                    df.loc[didx+i, 'squim_stoi_e'] = stoi_hyp.item()
                    df.loc[didx+i, 'squim_pesq_e'] = pesq_hyp.item()
                    df.loc[didx+i, 'squim_sisdr_e'] = si_sdr_hyp.item()    

                    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(ds_noisy)
                    mos = subjective_model(ds_noisy, nmr_speech)
                    df.loc[didx+i, 'squim_mos_n'] = mos.item()
                    df.loc[didx+i, 'squim_stoi_n'] = stoi_hyp.item()
                    df.loc[didx+i, 'squim_pesq_n'] = pesq_hyp.item()
                    df.loc[didx+i, 'squim_sisdr_n'] = si_sdr_hyp.item()    

                    df.loc[didx+i, 'squim_mos_i'] = df.loc[didx+i, 'squim_mos_e'] - df.loc[didx+i, 'squim_mos_n']
                    df.loc[didx+i, 'squim_stoi_i'] =  df.loc[didx+i, 'squim_stoi_e'] - df.loc[didx+i, 'squim_stoi_n']
                    df.loc[didx+i, 'squim_pesq_i'] = df.loc[didx+i, 'squim_pesq_e'] - df.loc[didx+i, 'squim_pesq_n']
                    df.loc[didx+i, 'squim_sisdr_i'] = df.loc[didx+i, 'squim_sisdr_e'] - df.loc[didx+i, 'squim_sisdr_n']


                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SQUIM of '+os.path.join(*meta[0][i].split('/')[6:])+'__'+model_name)
       
        df.to_csv(pjoin(results_path, model_name+'_evaluatedOn_Speakerphone.csv'), index=False)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Done.')

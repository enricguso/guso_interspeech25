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
import NISQA.nisqa.NISQA_lib as NL
from evaluation_metrics.nisqa_utils import load_nisqa_model
from evaluation_metrics.nisqa_utils import predict_nisqa
import evaluation_metrics.calculate_phoneme_similarity as phon
from evaluation_metrics.calculate_phoneme_similarity import LevenshteinPhonemeSimilarity
from espnet2.bin.spk_inference import Speech2Embedding
import evaluation_metrics.calculate_speaker_similarity as spksim

from discrete_speech_metrics import SpeechBERTScore
import evaluation_metrics.calculate_speechbert_score as sbert

#import evaluation_metrics.calculate_wer as wer

#HELPER FUCTIONS
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
    
    
class DFN_dataset(Dataset):
    def __init__(self, speech_path, noise_path, rir_path, n_samples):
        # Keep in mind that we should re-set the seeds if using any random process beneath
        # the gaussian noise generation for silent noise utterances

        # we store the textfile path in the class
        print('Initializing dataset...')
        self.speech_path = speech_path
        self.noise_path = noise_path
        self.rir_path = rir_path
        
        # load speech wav paths from the textfile
        self.speech_paths = []
        with open(speech_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                self.speech_paths.append(line.rstrip()) 
        print('speech set loaded. contains '+str(len(self.speech_paths)) +' files.')
        np.random.seed(0)


        self.speech_paths = np.random.choice(self.speech_paths, n_samples)
        print('selecting '+str(n_samples)+' files from that set')

        '''
        errors = []
        with open('dns_test_errors.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                errors.append(line.rstrip()) 
        self.speech_paths = [item for item in self.speech_paths if item not in errors]
        '''
        # we filter out all speech that does not come from read_speech
        #self.speech_paths = [item for item in self.speech_paths if item.split('/')[7]=='read_speech']
        
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
        # GENERATE THE CLEAN/NOISY PAIR
        clean = load_audio(self.speech_paths[idx])
        # handle weird corrupt speech path
        if len(clean) >= FS*DURATION:
            speech_nrgy = torch.mean(clean[:FS]**2) #we only check first second on long files, for speed purpose
        else:
            speech_nrgy = torch.mean(clean **2)
        if speech_nrgy == 0:
            clean = load_audio(self.speech_paths[0])

        noise = load_audio(self.noise_paths[idx])

        # handle corrupt rir
        try:
            rir = load_audio(self.rir_paths[idx])
        except:
            rir = torch.zeros(FS)
            rir[300] = 1.
        # handle silent rir
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
        lag = np.where(np.abs(rir) >= 0.5*np.max(np.abs(rir)))[0][0] # we take as direct sound the first value (from the left) that's at most -6dB from max

        revspeech = revspeech[lag:FS*DURATION + lag]

        # enforce energy conservation
        revspeech *= np.sqrt(power(clean) / power(revspeech)) 

        noisy = revspeech + noise
        
        # check for Nans
        if np.any(np.isnan(noisy)):
            print('noisy nan')
        if np.any(np.isnan(clean)):
            print('clean nan')
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)
        meta = [self.speech_paths[idx], self.noise_paths[idx], self.rir_paths[idx], self.snrs[idx].item()]
        return noisy.float(), clean.float(), meta
# dataset check
if __name__ == '__main__':
    FS = 48000
    DURATION = 4 #time in seconds of the eval chunk
    n_samples = 10000 #number of evaluation samples, originally 15000, 397 for rebuttal

    # names of the model folders (checkpoints/..) and aliases
    TRAINRIR_NAMES = {'D01_sb_none_NH_mono': 'singleband' , 'D02_mb_none_NH_mono': 'multiband', 
                'D03_mb_rec_NH_left': 'recdirectivity', 'D05_mb_srcrec_NH_left': 'recsourcedirectivity',
                'D00_DNS5': 'DNS5', 'D09_SSmp3d_left' : 'soundspaces'}
    use_gpu = True
    if torch.cuda.is_available() and use_gpu:
        TORCH_DEVICE = "cuda"
    else:
        TORCH_DEVICE = "cpu"

    num_workers = 8
    speech_path = '/home/ubuntu/Data/DFN/textfiles/readspeech_set.txt' #only read speech
    noise_path = '/home/ubuntu/Data/DFN/textfiles/test_set_noise.txt'
    dns_mos_path = '/home/ubuntu/enric/DNS-Challenge/DNSMOS/DNSMOS'

    #'/home/ubuntu/Data/DFN/textfiles/multiband_test_rir.txt',
    #'/home/ubuntu/Data/DFN/textfiles/singleband_test_rir.txt',
    #'/home/ubuntu/Data/DFN/textfiles/recdirectivity_left_test_rir.txt',
    #'/home/ubuntu/Data/DFN/textfiles/recsourcedirectivity_left_test_rir.txt']
    rir_path = '/home/ubuntu/Data/DFN/textfiles/real_rirs.txt'
    model_names = list(TRAINRIR_NAMES.keys())
    results_path = 'results_waspaa'

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

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    objective_model = SQUIM_OBJECTIVE.get_model()

    subjective_model = subjective_model.to(TORCH_DEVICE)
    objective_model = objective_model.to(TORCH_DEVICE)

    NMR_SPEECH = torchaudio.utils.download_asset("tutorial-assets/ctc-decoding/1688-142285-0007.wav")
    nmr_speech, _ = torchaudio.load(NMR_SPEECH)
    nmr_speech = nmr_speech.to(TORCH_DEVICE)

    nisqa_model = load_nisqa_model('NISQA/weights/nisqa.tar', 'cuda')
    phon_model = LevenshteinPhonemeSimilarity(device='cuda')
    phon_model.phoneme_predictor.eval();
    spksim_model = Speech2Embedding.from_pretrained(
        model_tag = "espnet/voxcelebs12_rawnet3", device='cuda')
    spksim_model.spk_model.eval();
    sbs_model = SpeechBERTScore()
    sbs_model.model.eval();

    dataset = DFN_dataset(speech_path, noise_path, rir_path, n_samples) 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=True) 

    for model_name in model_names: #for each model (or set of training RIRs)
        # initialize CSV file
        df = pd.DataFrame(columns=['train_rirs', 'eval_rirs', 'model', 'speech', 'noise', 'rir', 'noisy_snr', 
                            # ENHANCED METRICS
                            'sisdr_e', 'pesq_e', 'stoi_e', #'srmr_e', 
                            'dnsmos_ovrl_e', 'dnsmos_sig_e', 'dnsmos_bak_e', 'dnsmos_p808_e',
                            'squim_mos_e', 'squim_stoi_e', 'squim_pesq_e', 'squim_sisdr_e',
                            'estoi_e', 'mcd_e', 'pesq2_e', 'sdr_e', 'nisqa_mos_e', 'nisqa_noi_e', 'nisqa_dis_e', 'nisqa_col_e',
                            'nisqa_loud_e', 'phonsim_e', 'spksim_e', 'sBertSim_e',
                            # CLEAN METRICS
                            #'sisdr_c', 'pesq_c', 'stoi_c', 'srmr_c', 
                            'dnsmos_ovrl_c', 'dnsmos_sig_c', 'dnsmos_bak_c', 'dnsmos_p808_c',#'srmr_c', 
                            'squim_mos_c', 'squim_stoi_c', 'squim_pesq_c', 'squim_sisdr_c',
                            #'estoi_c', 'mcd_c', 'pesq2_c', 'sdr_c', 
                            'nisqa_mos_c', 'nisqa_noi_c', 'nisqa_dis_c', 'nisqa_col_c',
                            'nisqa_loud_c', #'phonsim_c', 'spksim_c',
                            # NOISY METRICS
                            'sisdr_n', 'pesq_n', 'stoi_n', #'srmr_n', 
                            'dnsmos_ovrl_n', 'dnsmos_sig_n', 'dnsmos_bak_n', 'dnsmos_p808_n',
                            'squim_mos_n', 'squim_stoi_n', 'squim_pesq_n', 'squim_sisdr_n',
                            'estoi_n', 'mcd_n', 'pesq2_n', 'sdr_n', 'nisqa_mos_n', 'nisqa_noi_n', 'nisqa_dis_n', 'nisqa_col_n',
                            'nisqa_loud_n', 'phonsim_n', 'spksim_n', 'sBertSim_n',
                            # INCREMENT (ENHANCED - NOISY)
                            'sisdr_i', 'pesq_i', 'stoi_i', #'srmr_i', 
                            'dnsmos_ovrl_i', 'dnsmos_sig_i', 'dnsmos_bak_i', 'dnsmos_p808_i',
                            'squim_mos_i', 'squim_stoi_i', 'squim_pesq_i', 'squim_sisdr_i',
                            'estoi_i', 'mcd_i', 'pesq2_i', 'sdr_i', 'nisqa_mos_i', 'nisqa_noi_i', 'nisqa_dis_i', 'nisqa_col_i',
                            'nisqa_loud_i', 'phonsim_i', 'spksim_i', 'sBertSim_i'
                            ])

        model_path = pjoin('/home/ubuntu/Data/DFN', model_name)
        model, df_state, _ = init_df(model_path)
        
        print('Checking data integrity...')
        for noisy, clean, meta in tqdm.tqdm(dataloader):
            continue
            # if this crashes there's a problem with the data or the dataloader
        print('Done.')
        # TODO ENDS ##############################################

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Running evaluation of '+model_name+' evaluated on '+
            rir_path.split('/')[-1].split('_')[0]+'RIRs ...')
        
        for noisy, clean, meta in tqdm.tqdm(dataloader):
            try:
                enhanced = enhance(model, df_state, noisy)
            except:
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error while enhancing '+os.path.join(*meta[0][0].split('/')[6:]))

            downsampler = downsampler.to('cpu')
            ds_enhanced = downsampler(enhanced)
            ds_clean = downsampler(clean)
            ds_noisy = downsampler(noisy)

            didx = len(df)
            for i in range(1): # this could be removed if we stick to batch size 1
                df.loc[didx+i, 'train_rirs'] = TRAINRIR_NAMES[model_name]
                df.loc[didx+i, 'eval_rirs'] = rir_path.split('/')[-1].split('_')[0]
                df.loc[didx+i, 'model'] = model_name
                df.loc[didx+i, 'speech'] = os.path.join(*meta[0][i].split('/')[6:])
                df.loc[didx+i, 'noise'] = os.path.join(*meta[1][i].split('/')[6:])
                df.loc[didx+i, 'rir'] = os.path.join(*meta[2][i].split('/')[4:])
                df.loc[didx+i, 'noisy_snr'] = meta[3][i].item()
                
                try:
                # first the DNSMOS metrics
                    dnsmos_result_e = dnsmos.run(ds_enhanced[i].numpy(), 16000) 
                    dnsmos_result_n = dnsmos.run(ds_noisy[i].numpy(), 16000) 
                    dnsmos_result_c = dnsmos.run(ds_clean[i].numpy(), 16000) 

                    df.loc[didx+i, 'dnsmos_ovrl_e'] = dnsmos_result_e['ovrl_mos']
                    df.loc[didx+i, 'dnsmos_sig_e'] = dnsmos_result_e['sig_mos']
                    df.loc[didx+i, 'dnsmos_bak_e'] = dnsmos_result_e['bak_mos']
                    df.loc[didx+i, 'dnsmos_p808_e'] = dnsmos_result_e['p808_mos']

                    df.loc[didx+i, 'dnsmos_ovrl_c'] = dnsmos_result_c['ovrl_mos']
                    df.loc[didx+i, 'dnsmos_sig_c'] = dnsmos_result_c['sig_mos']
                    df.loc[didx+i, 'dnsmos_bak_c'] = dnsmos_result_c['bak_mos']
                    df.loc[didx+i, 'dnsmos_p808_c'] = dnsmos_result_c['p808_mos']

                    df.loc[didx+i, 'dnsmos_ovrl_n'] = dnsmos_result_n['ovrl_mos']
                    df.loc[didx+i, 'dnsmos_sig_n'] = dnsmos_result_n['sig_mos']
                    df.loc[didx+i, 'dnsmos_bak_n'] = dnsmos_result_n['bak_mos']
                    df.loc[didx+i, 'dnsmos_p808_n'] = dnsmos_result_n['p808_mos']

                    df.loc[didx+i, 'dnsmos_ovrl_i'] = dnsmos_result_e['ovrl_mos'] - dnsmos_result_n['ovrl_mos']
                    df.loc[didx+i, 'dnsmos_sig_i'] = dnsmos_result_e['sig_mos'] - dnsmos_result_n['sig_mos']
                    df.loc[didx+i, 'dnsmos_bak_i'] = dnsmos_result_e['bak_mos'] - dnsmos_result_n['bak_mos']
                    df.loc[didx+i, 'dnsmos_p808_i'] = dnsmos_result_e['p808_mos'] - dnsmos_result_n['p808_mos']
                    
                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting DNSMOS of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)

                    df.loc[didx+i, 'dnsmos_ovrl_e'] = np.NAN
                    df.loc[didx+i, 'dnsmos_sig_e'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_bak_e'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_p808_e'] =  np.NAN

                    df.loc[didx+i, 'dnsmos_ovrl_c'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_sig_c'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_bak_c'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_p808_c'] =  np.NAN

                    df.loc[didx+i, 'dnsmos_ovrl_n'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_sig_n'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_bak_n'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_p808_n'] =  np.NAN

                    df.loc[didx+i, 'dnsmos_ovrl_i'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_sig_i'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_bak_i'] =  np.NAN
                    df.loc[didx+i, 'dnsmos_p808_i'] =  np.NAN
                #now the URGNT METRICS
                try:
                    df.loc[didx+i, 'estoi_e'] = intru.estoi_metric(clean.squeeze(), enhanced.squeeze(), FS)
                    df.loc[didx+i, 'estoi_n'] = intru.estoi_metric(clean.squeeze(), noisy.squeeze(), FS)
                    df.loc[didx+i, 'estoi_i'] = df.loc[didx+i, 'estoi_e'] - df.loc[didx+i, 'estoi_n']
                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting ESTOI of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                try:
                    df.loc[didx+i, 'lsd_e'] = intru.lsd_metric(clean.numpy(), enhanced.numpy(), FS)[0]
                    df.loc[didx+i, 'lsd_n'] = intru.lsd_metric(clean.numpy(), noisy.numpy(), FS)[0]
                    df.loc[didx+i, 'lsd_i'] = df.loc[didx+i, 'lsd_e'] - df.loc[didx+i, 'lsd_n'] #this will give -increment

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting LSD of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                try:
                    df.loc[didx+i, 'mcd_e'] = intru.mcd_metric(clean.squeeze().numpy(), enhanced.squeeze().numpy(), FS)
                    df.loc[didx+i, 'mcd_n'] = intru.mcd_metric(clean.squeeze().numpy(), noisy.squeeze().numpy(), FS)
                    df.loc[didx+i, 'mcd_i'] = df.loc[didx+i, 'mcd_e'] - df.loc[didx+i, 'mcd_n']  #this will give -increment
                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting MCD of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                try:
                    df.loc[didx+i, 'pesq2_e'] = intru.pesq_metric(clean.squeeze().numpy(), enhanced.squeeze().numpy(), FS)
                    df.loc[didx+i, 'pesq2_n'] = intru.pesq_metric(clean.squeeze().numpy(), noisy.squeeze().numpy(), FS)
                    df.loc[didx+i, 'pesq2_i'] = df.loc[didx+i, 'pesq2_e'] - df.loc[didx+i, 'pesq2_n']

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting PESQ2 of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                try:
                    df.loc[didx+i, 'sdr_e'] = intru.sdr_metric(clean.squeeze().numpy(), enhanced.squeeze().numpy())
                    df.loc[didx+i, 'sdr_n'] = intru.sdr_metric(clean.squeeze().numpy(), noisy.squeeze().numpy())
                    df.loc[didx+i, 'sdr_i'] = df.loc[didx+i, 'sdr_e'] - df.loc[didx+i, 'sdr_n']

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SDR of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                try:
                    torchaudio.save('tmp.wav', enhanced, FS)
                    nisqa_sco = predict_nisqa(nisqa_model, 'tmp.wav')
                    df.loc[didx+i, 'nisqa_mos_e'] = nisqa_sco['mos_pred']
                    df.loc[didx+i, 'nisqa_noi_e'] = nisqa_sco['noi_pred']
                    df.loc[didx+i, 'nisqa_dis_e'] = nisqa_sco['dis_pred']
                    df.loc[didx+i, 'nisqa_col_e'] = nisqa_sco['col_pred']
                    df.loc[didx+i, 'nisqa_loud_e'] = nisqa_sco['loud_pred']

                    torchaudio.save('tmp.wav', clean, FS)
                    nisqa_sco = predict_nisqa(nisqa_model, 'tmp.wav')
                    df.loc[didx+i, 'nisqa_mos_c'] = nisqa_sco['mos_pred']
                    df.loc[didx+i, 'nisqa_noi_c'] = nisqa_sco['noi_pred']
                    df.loc[didx+i, 'nisqa_dis_c'] = nisqa_sco['dis_pred']
                    df.loc[didx+i, 'nisqa_col_c'] = nisqa_sco['col_pred']
                    df.loc[didx+i, 'nisqa_loud_c'] = nisqa_sco['loud_pred']

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
                try:
                    # from -inf to 1
                    df.loc[didx+i, 'phonsim_e'] = phon.phoneme_similarity_metric(phon_model, clean.squeeze(), enhanced.squeeze(), FS)
                    df.loc[didx+i, 'phonsim_n'] = phon.phoneme_similarity_metric(phon_model, clean.squeeze(), noisy.squeeze(), FS)
                    df.loc[didx+i, 'phonsim_i'] = df.loc[didx+i, 'phonsim_e'] - df.loc[didx+i, 'phonsim_n']

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting PhonSim of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                try:
                    df.loc[didx+i, 'spksim_e'] = spksim.speaker_similarity_metric(spksim_model, clean.squeeze(), enhanced.squeeze(), FS)
                    df.loc[didx+i, 'spksim_n'] = spksim.speaker_similarity_metric(spksim_model, clean.squeeze(), noisy.squeeze(), FS)
                    df.loc[didx+i, 'spksim_i'] = df.loc[didx+i, 'spksim_e'] - df.loc[didx+i, 'spksim_n']

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SpkSim of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                try:
                    df.loc[didx+i, 'sBertSim_e'] = sbert.speech_bert_score_metric(sbs_model.score, clean.squeeze(), enhanced.squeeze(), FS)
                    df.loc[didx+i, 'sBertSim_n'] = sbert.speech_bert_score_metric(sbs_model.score, clean.squeeze(), noisy.squeeze(), FS)
                    df.loc[didx+i, 'sBertSim_i'] = df.loc[didx+i, 'sBertSim_e'] - df.loc[didx+i, 'sBertSim_n']

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SpeechBERTScore of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)             


            # then to GPU
            ds_enhanced = ds_enhanced.to(TORCH_DEVICE)
            ds_clean = ds_clean.to(TORCH_DEVICE)
            ds_noisy = ds_noisy.to(TORCH_DEVICE)
            noisy = noisy.to(TORCH_DEVICE)
            clean = clean.to(TORCH_DEVICE)
            enhanced = enhanced.to(TORCH_DEVICE)
            for i in range(1):
                try:
                    #SQUIM METRICS
                    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(ds_enhanced)
                    mos = subjective_model(ds_enhanced, nmr_speech)
                    df.loc[didx+i, 'squim_mos_e'] = mos.item()
                    df.loc[didx+i, 'squim_stoi_e'] = stoi_hyp.item()
                    df.loc[didx+i, 'squim_pesq_e'] = pesq_hyp.item()
                    df.loc[didx+i, 'squim_sisdr_e'] = si_sdr_hyp.item()    

                    stoi_hyp, pesq_hyp, si_sdr_hyp = objective_model(ds_clean)
                    mos = subjective_model(ds_clean, nmr_speech)
                    df.loc[didx+i, 'squim_mos_c'] = mos.item()
                    df.loc[didx+i, 'squim_stoi_c'] = stoi_hyp.item()
                    df.loc[didx+i, 'squim_pesq_c'] = pesq_hyp.item()
                    df.loc[didx+i, 'squim_sisdr_c'] = si_sdr_hyp.item()   

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
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SQUIM of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)

                try:
                    sisdr_e = sisdr_single(enhanced[i], clean[i]).item() 
                    sisdr_n = sisdr_single(noisy[i], clean[i]).item() 
                    sisdr_i = sisdr_e - sisdr_n
                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting SISDR of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)                    
                    sisdri_e = np.NAN
                    sisdr_n = np.NAN
                    sisdri = np.NAN
                df.loc[didx+i, 'sisdr_e'] = sisdr_e
                df.loc[didx+i, 'sisdr_n'] = sisdr_n
                df.loc[didx+i, 'sisdr_i'] = sisdr_i

                try:
                    df.loc[didx+i, 'pesq_e'] = pesq_single(ds_enhanced[i], ds_clean[i]).item()
                    df.loc[didx+i, 'pesq_n'] = pesq_single(ds_noisy[i], ds_clean[i]).item()
                    df.loc[didx+i, 'pesq_i'] = df.loc[didx+i, 'pesq_e'] - df.loc[didx+i, 'pesq_n']

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting PESQ of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)
                    df.loc[didx+i, 'pesq_e'] = np.NAN
                    df.loc[didx+i, 'pesq_n'] = np.NAN
                    df.loc[didx+i, 'pesq_i'] = np.NAN

                try:
                    df.loc[didx+i, 'stoi_e'] = stoi_single(enhanced[i], clean[i]).item()
                    df.loc[didx+i, 'stoi_n'] = stoi_single(noisy[i], clean[i]).item()
                    df.loc[didx+i, 'stoi_i'] = df.loc[didx+i, 'stoi_e'] - df.loc[didx+i, 'stoi_n']

                except:
                    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Error getting STOI of '+os.path.join(*meta[0][i].split('/')[6:])+'_'+rir_path+'_'+model_name)                    
                    df.loc[didx+i, 'stoi_e'] = np.NAN
                    df.loc[didx+i, 'stoi_n'] = np.NAN
                    df.loc[didx+i, 'stoi_i'] = np.NAN                   

        df.to_csv(pjoin(results_path, model_name+'_evaluatedOn_'+rir_path.split('/')[-1].split('_')[0]+'_drynoise.csv'), index=False)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+' || Done.')

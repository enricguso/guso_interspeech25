import masp as srs
import numpy as np
import soundfile as sf
import scipy
import copy
import pandas as pd
import os
from os.path import join as pjoin
import mat73
import scipy.signal as sig
import argparse 
from datetime import datetime

def load_speechdirectivity(path, plot):
    # load and parse average human speech directivity data
    dirdata = scipy.io.loadmat(path)['azel_dir']
    d = {}
    bands = np.array(['100Hz', '125Hz', '160Hz', '200Hz', '250Hz', '315Hz', '400Hz', '500Hz', '630Hz', '800Hz', '1000Hz', '1250Hz', '1600Hz', '2000Hz', '2500Hz', '3150Hz', '4000Hz', '5000Hz', '6300Hz', '8000Hz', '10000Hz'])
    for i, band in enumerate(bands):
        d[band] = dirdata[i]
    az_axis = np.linspace(-180, 175, 72)
    el_axis = np.linspace(-90, 85, 36)
    d['az_axis'] = az_axis
    d['el_axis'] = el_axis
    if plot:
        plt.figure(figsize=(12,20))
        for i, band in enumerate(bands):
            plt.subplot(7,3,i+1)
            plt.imshow(d[band], cmap='jet')
            plt.yticks(range(len(d['el_axis']))[::8], [int(x) for x in d['el_axis'][::8]], fontsize=7)
            plt.ylabel('elevation')
            plt.xticks(range(len(d['az_axis']))[::8], [int(x) for x in d['az_axis'][::8]], rotation=90, fontsize=7)
            plt.xlabel('azimuth')
            plt.title(band)
            plt.clim(-20,0)
        cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.72])  # [left, bottom, width, height]
        cbar = plt.colorbar(cax=cbar_ax)
        plt.savefig('speech_directivity.pdf')
    return d

def get_6band_rt60_vector():
    # Check 'notebooks/RT60_analysis_AEC.ipynb' for more info
    np.random.seed() #we randomize so multiprocessing doesn't yield same RT60s
    alphas = np.array([1.7196874268124676,
                         1.6152228672267106,
                         1.9318203836226113,
                         2.55718115999814,
                         4.176814897493042,
                         2.4892656080814346])
    betas = np.array([0.38685390302225775,
                         0.24453641709737417,
                         0.14321372785643122,
                         0.10453218827453133,
                         0.08678871224845529,
                         0.18290733668646034])
    sim_rt60Fs = []
    for i in range(len(alphas)):
        sim_rt60Fs.append(np.random.gamma(alphas[i], betas[i], 1))
    return np.array(sim_rt60Fs).squeeze()
    
def head_2_ku_ears(head_pos,head_orient):
# based on head pos and orientation, compute coordinates of ears
    ear_distance_ku100=0.0875
    theta = (head_orient[0]) * np.pi / 180
    R_ear = [head_pos[0] - ear_distance_ku100 * np.sin(theta),
              head_pos[1] + ear_distance_ku100 * np.cos(theta), 
              head_pos[2]]
    L_ear = [head_pos[0] + ear_distance_ku100 * np.sin(theta),
              head_pos[1] - ear_distance_ku100 * np.cos(theta), 
              head_pos[2]]
    return [L_ear,R_ear]
    
def plot_scene(room_dims,head_pos,head_orient,l_mic_pos,l_src_pos, perspective="xy"):
#   function to plot the designed scene
#   room_dims - dimensions of the room [x,y,z]
#   head_pos - head position [x,y,z]
#   head_orient - [az,el]
#   l_src_pos - list of source positions [[x,y,z],...,[x,y,z]]
#   ref_pos - reflection to plot position
#   perspective - which two dimensions to show 
    if perspective=="xy":
        dim1=1
        dim2=0
    elif perspective=="yz":
        dim1=2
        dim2=1
    elif perspective=="xz":
        dim1=2
        dim2=0
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim((0,room_dims[dim1]))
    plt.ylim((0,room_dims[dim2]))
    plt.axvline(head_pos[dim1], color='y') # horizontal lines
    plt.axhline(head_pos[dim2], color='y') # vertical lines
    plt.grid(True)
    # plot sources and receivers
    plt.plot(head_pos[dim1],head_pos[dim2], "o", ms=10, mew=2, color="black")
    # plot ears
    plt.plot(l_mic_pos[0][dim1],l_mic_pos[0][dim2], "o", ms=3, mew=2, color="blue")# left ear in blue
    plt.plot(l_mic_pos[1][dim1],l_mic_pos[1][dim2], "o", ms=3, mew=2, color="red")# right ear in red

    for i,src_pos in enumerate(l_src_pos):
        plt.plot(src_pos[dim1],src_pos[dim2], "o", ms=10, mew=2, color="red")
        plt.annotate(str(i), (src_pos[dim1],src_pos[dim2]))

    # plot head orientation if looking from above 
    if perspective=="xy":
        plt.plot(head_pos[dim1],head_pos[dim2], marker=(1, 1, -head_orient[0]), ms=20, mew=2,color="black")

    ax.set_aspect('equal', adjustable='box')
    
def add_azi(ang1, ang2):
    # ang1 and ang2 are defined from [-180 to 180]
    result = ang1 + ang2
    if result < -180:
        result += 360
    elif  result > 180:
        result -= 360   
    if result == 180:
        result = -180
    return result

def add_ele(ang1, ang2):
    # ang1 and ang2 are defined from [-90 to 90]
    result = ang1 + ang2
    if result < -90:
        result += 180
    elif  result > 90:
        result -= 180   
    if result == 90:
        result = -90
    return result
    
def apply_directivity(echograms, recip_echograms, sourceOrient, d, band_centerfreqs):
    
    # echograms: absorption echograms with shape [source, receiver, band]
    # recip_echograms: absorption reciprocal echograms with shape [receiver, source, band]
    # sourceOrient: clockwise source rotation in degrees, from -180 to 180 azimuth and -90 to 90 elevation, [azi, el]
    # d: average speech directivity dict, with d['100Hz'].shape == [36 (elevation),72(azimuth)]
    nSrc = echograms.shape[0]
    nRec = echograms.shape[1]
    directivity_echograms = copy.deepcopy(echograms)
    for ns in range(nSrc): # for each source
        for nr in range(nRec): # for each receiver 
            for bd in range(echograms.shape[2]): # for each band
                band = str(int(band_centerfreqs[bd]))+'Hz'
                # pass reciprocal coordinates to azimuth elevation
                sph = srs.utils.cart2sph(recip_echograms[nr, ns, bd].coords)
                azi = 180 * sph[:,0] / np.pi
                ele = 180 * sph[:,1] / np.pi # rad to degrees
                #qazi = np.zeros_like(azi)
                azi_rot = np.zeros_like(azi)
                azi_idxs = np.zeros_like(azi).astype(int)
                for i in range(len(azi)):
                    azi_rot[i] = add_azi(azi[i], sourceOrient[0])
                    azi_idxs[i] = int(np.argmin(np.abs(d['az_axis'] - azi_rot[i])))
                    #qazi[i] = d['az_axis'][azi_idxs[i]]
                #qele = np.zeros_like(ele)
                ele_rot = np.zeros_like(ele)
                ele_idxs = np.zeros_like(ele).astype(int)
                for i, angle in enumerate(ele):
                    ele_rot[i] = add_azi(ele[i], sourceOrient[1])
                    ele_idxs[i] = int(np.argmin(np.abs(d['el_axis'] - ele_rot[i])))
                    #qele[i] = d['el_axis'][ele_idxs[i]]
                factors = np.zeros(len(directivity_echograms[ns,nr,bd].value))
                for r in range(len(factors)):
                    factors[r] = np.power(10, d[band][ele_idxs[r], azi_idxs[r]] / 20)
                factors = np.tile(factors, (directivity_echograms[ns,nr,bd].value.shape[1], 1)).T
                directivity_echograms[ns, nr, bd].value *= factors
        print('directivity applied')
    return directivity_echograms

# This script encapsulated in a method for multi-processing takes a dataframe row and stores the audio on disk
# a = df.iloc[i]
def process(a):
    try:

        headC = np.array([a.headC_x, a.headC_y, a.headC_z])
        headOrient = np.array([a.head_azi, a.head_ele])
        src = np.array([[a.src_x, a.src_y, a.src_z]])
        srcOrient = np.array([a.src_azi, a.src_ele])
        room = np.array([a.room_x, a.room_y, a.room_z])
        
        rt60 = np.array([a.rt60])
        
        rt60s = np.array([a.rt60_125hz, a.rt60_250hz, a.rt60_500hz, a.rt60_1000hz, a.rt60_2000hz, a.rt60_4000hz])
        #speech, fs_speech = lsa.load('ane_speech.wav', sr=fs_rir)
        mic = np.array(head_2_ku_ears(headC,headOrient)) # we get BiMagLS mic points 
        mic = np.vstack((mic, headC)) # we add the head center microphone for non binaural decoders

        
        abs_walls,_ = srs.find_abs_coeffs_from_rt(room, rt60s)
        abs_walls_single, _ = srs.find_abs_coeffs_from_rt(room, rt60)
        limits_single = np.minimum(rt60, maxlim)
        limits = np.minimum(rt60s, maxlim)
        print('limits', limits)
        echograms_single = srs.compute_echograms_mic(room, src, np.array([mic[2]]), abs_walls_single, limits_single, np.array([[1,0,0,1]]))
        print('single echogram')
        echograms_mb = srs.compute_echograms_mic(room, src, np.array([mic[2]]), abs_walls, limits, np.array([[1,0,0,1]]))
        print('mb echogram')
        echograms_sh  = srs.compute_echograms_sh(room, src, mic[0:2], abs_walls, limits, ambi_order, headOrient)
        print('SH echograms')
        recip_echograms  = srs.compute_echograms_sh(room, mic[0:2], src, abs_walls, limits, ambi_order, headOrient)
        print('rec echograms computed')
        directivity_echograms = apply_directivity(echograms_sh, recip_echograms, srcOrient, d, band_centerfreqs)
        print('src rec echograms')
        mic_rirs_single = srs.render_rirs_mic(echograms_single, np.array([1000]), fs_rir)
        mic_rirs_mb = srs.render_rirs_mic(echograms_mb, band_centerfreqs, fs_rir)
        recdir_rirs = srs.render_rirs_sh(echograms_sh, band_centerfreqs, fs_rir)#/np.sqrt(4*np.pi)  
        recsrcdir_rirs = srs.render_rirs_sh(directivity_echograms, band_centerfreqs, fs_rir)#/np.sqrt(4*np.pi)
        print('SH rirs')
        
        bin_ir_recdir = np.array([sig.fftconvolve(np.squeeze(recdir_rirs[:,:,0, 0]), decoder[:,:,0], 'full', 0).sum(1),
                        sig.fftconvolve(np.squeeze(recdir_rirs[:,:,1, 0]), decoder[:,:,1], 'full', 0).sum(1)])
        
        bin_ir_recsrcdir = np.array([sig.fftconvolve(np.squeeze(recsrcdir_rirs[:,:,0, 0]), decoder[:,:,0], 'full', 0).sum(1),
                        sig.fftconvolve(np.squeeze(recsrcdir_rirs[:,:,1, 0]), decoder[:,:,1], 'full', 0).sum(1)])
        bin_ir_recsrcdirHA = np.array([sig.fftconvolve(np.squeeze(recsrcdir_rirs[:,:,0, 0]), decoderHA[:,:,0], 'full', 0).sum(1),
                        sig.fftconvolve(np.squeeze(recsrcdir_rirs[:,:,1, 0]), decoderHA[:,:,1], 'full', 0).sum(1)])
        print('bin rirs')
        single_max = np.max(np.abs(mic_rirs_single))
        mb_max = np.max(np.abs(mic_rirs_mb))
        recdir_max = np.max(np.abs(bin_ir_recdir))
        recsrcdir_max = np.max(np.abs(bin_ir_recsrcdir))
        recsrcdirHA_max = np.max(np.abs(bin_ir_recsrcdirHA))
        oallmax = np.max((single_max, mb_max, recdir_max, recsrcdir_max, recsrcdirHA_max))
        if oallmax >= 0.95:
            oallmax = 0.95
        mic_rirs_single /= single_max
        mic_rirs_single *= oallmax
        mic_rirs_mb /= mb_max
        mic_rirs_mb *= oallmax
        bin_ir_recdir /= recdir_max
        bin_ir_recdir *= oallmax
        bin_ir_recsrcdir /= recsrcdir_max
        bin_ir_recsrcdir *= oallmax
        bin_ir_recsrcdirHA /= recsrcdirHA_max
        bin_ir_recsrcdirHA *= oallmax

        print('normalized')
        sing_path = pjoin(pjoin(pjoin(output_path, a.set), 'singleband'), "{:05d}".format(a.id) + '.wav')
        mb_path = pjoin(pjoin(pjoin(output_path, a.set), 'multiband'), "{:05d}".format(a.id) + '.wav')
        recleft_path = pjoin(pjoin(pjoin(output_path, a.set), 'recdirectivity_left'), "{:05d}".format(a.id) + '.wav')
        recright_path = pjoin(pjoin(pjoin(output_path, a.set), 'recdirectivity_right'), "{:05d}".format(a.id) + '.wav')
        recsrcleft_path = pjoin(pjoin(pjoin(output_path, a.set), 'recsourcedirectivity_left'), "{:05d}".format(a.id) + '.wav')
        recsrcright_path = pjoin(pjoin(pjoin(output_path, a.set), 'recsourcedirectivity_right'), "{:05d}".format(a.id) + '.wav')
        
        recsrcleftHA_path = pjoin(pjoin(pjoin(output_path, a.set), 'recsourcedirectivityHA_left'), "{:05d}".format(a.id) + '.wav')        
        recsrcrightHA_path = pjoin(pjoin(pjoin(output_path, a.set), 'recsourcedirectivityHA_right'), "{:05d}".format(a.id) + '.wav')

        sf.write(sing_path, mic_rirs_single[:,0,0], fs_rir, subtype='FLOAT')   
        sf.write(mb_path, mic_rirs_mb[:,0,0], fs_rir, subtype='FLOAT')    
        sf.write(recleft_path, bin_ir_recdir[0], fs_rir, subtype='FLOAT')    
        sf.write(recright_path, bin_ir_recdir[1], fs_rir, subtype='FLOAT')    
        sf.write(recsrcleft_path, bin_ir_recsrcdir[0], fs_rir, subtype='FLOAT')    
        sf.write(recsrcright_path, bin_ir_recsrcdir[1], fs_rir, subtype='FLOAT')   
        sf.write(recsrcleftHA_path, bin_ir_recsrcdirHA[0], fs_rir, subtype='FLOAT')    
        sf.write(recsrcrightHA_path, bin_ir_recsrcdirHA[1], fs_rir, subtype='FLOAT')    
        print('written')
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print('File '+str(a.id)+ ' done. ' +formatted_time)
        print(' ')
    except:
        print('ERROR when processing ' + str(a.id))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Dataset Generation Argument Parser')
    parser.add_argument("--output", type=str,
                    help="""Directory where to save processed wavs.""",
                    default=None)
    parser.add_argument('--workers', type=int, default=20, 
                        help='Number of workers to be used (default is 20).')
    parser.add_argument('--cpu', type=int, default=0, 
                        help='Which CPU are we running')
    
    args = parser.parse_args()

    num_workers = args.workers 
    output_path = args.output

    d = load_speechdirectivity(path=pjoin('directivity_parsing_matlab', 'azel_dir.mat'), plot=False)
    band_centerfreqs = np.zeros((6))
    band_centerfreqs[0] = 125
    for nb in range(5):
        band_centerfreqs[nb+1] = 2 * band_centerfreqs[nb]

    decoder_path = pjoin('decoders_ord10', 'Ku100_ALFE_Window_sinEQ_bimag.mat') #10th order BimagLS decoder del KU100 sin HA a 48kHz
    decoder = mat73.loadmat(decoder_path)['hnm']
    decoder = np.roll(decoder,500,axis=0)

    decoder_pathHA = pjoin('decoders_ord10', 'RIC_Front_Omni_ALFE_Window_SinEQ_bimag.mat') #10th order BimagLS decoder del HA Amplifon a 48kHz
    decoderHA = mat73.loadmat(decoder_pathHA)['hnm']
    decoderHA = np.roll(decoder,500,axis=0)

    maxlim = 2 # maximum reflection time in seconds. Stop simulating if it goes beyond that time.
    ambi_order = 10 # ambisonics order
    fs_rir = 48000

    df_path = 'meta_ins25.csv'
    df = pd.read_csv(df_path)
    
    print('RIR dataset generation script. Interspeech2025.')

    # we select the dataset subset for that specific CPU
    idx = int(args.cpu * len(df)/args.workers) 
    print('CPU '+ str(args.cpu))
    df = df[idx : int(idx + len(df)/args.workers)]
    
    # make dirs if not exist
    sets = ['train', 'val', 'test']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for subset in sets:
        if not os.path.exists(pjoin(output_path, subset)):
            os.makedirs(pjoin(output_path, subset))
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'singleband')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'singleband'))
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'multiband')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'multiband'))
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'recdirectivity_left')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'recdirectivity_left'))
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'recdirectivity_right')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'recdirectivity_right'))
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'recsourcedirectivity_left')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'recsourcedirectivity_left'))
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'recsourcedirectivity_right')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'recsourcedirectivity_right'))  
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'recsourcedirectivityHA_left')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'recsourcedirectivityHA_left'))
        if not os.path.exists(pjoin(pjoin(output_path, subset), 'recsourcedirectivityHA_right')):
            os.makedirs(pjoin(pjoin(output_path, subset), 'recsourcedirectivityHA_right'))  
    
    # we remove the already processed files from the queue -- the metadata df
    already = os.listdir(pjoin(pjoin(output_path, 'train'),'recsourcedirectivityHA_right'))
    already += os.listdir(pjoin(pjoin(output_path, 'val'), 'recsourcedirectivityHA_right'))
    already += os.listdir(pjoin(pjoin(output_path, 'test'), 'recsourcedirectivityHA_right'))
    print(str(len(already))+' files already processed.')
    print(' ')
    already = [int(x.split('.wav')[0]) for x in already]
    df = df.drop(already, errors='ignore')
    # we shuffle to avoid getting stuck in the tricky rooms
    np.random.seed()
    permu = np.random.permutation(len(df))
    for i in  range(len(df)):
        print('starting to process ', i)
        process(df.iloc[permu[i]])
    print(' ')
    print('All files processed. Done.')

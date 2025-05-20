# guso-waspaa25
This software allows to reproduce our Waspaa 2025 submission.

Feel free to reach out to any of the following authors if needed:

>Enric Gusó enric.guso@eurecat.org

>Joanna Luberazdka joanna.luberadzka@eurecat.org

>Umut Sayin umut.sayin@eurecat.org

## Contents
The repository is structured into two main components:
* `notebooks/dataset_design.ipynb`: in case you want to re-generate `meta_was25.csv`, the metadata with all the acoustic parameters of the 60.000 RIRs: room dimensions, T60s, source and receiver positions and orientations, etc.
* `generate_data.py`: the main script that transforms the `meta_was25.csv` into `.wav` files impulse responses. Note that this won't run in a consumer-grade machine, we have used an AWS instance with 320GB of memory. Instead of using `multiprocessing`, we have split the RIRs to process with `--workers`and `--cpu` arguments and let the OS handle the resources, which means you should adjust both depending on your hadrware. For example, if your machine can handle 3 workers, you can run in parallel;

`python generate_data.py --output <you_output_path> --workers 3 --cpu 0`

`python generate_data.py --output <you_output_path> --workers 3 --cpu 1`

`python generate_data.py --output <you_output_path> --workers 3 --cpu 2`

Additionally , we provide:
* A SOFA file with 50-point pseudo-anechoic HRTFs of a KU100 dummy head: `sofa_files/KU100_New_128_noALFE_cut_now.sofa`
* A SOFA file with 50-point pseudo-anechoic HRTFs of a KU100 dummy head wearing an audifon lewi R HA RIC device: `sofa_files/RIC_Front_Omni_128_noALFE_cut_now.sofa`
* 10-th order Ambisonics to Binaural decoders for both SOFA files: `decoders_ord10/RIC_Front_Omni.mat` and `decoders_ord10/Ku100_ALFE_Window_sinEQ_bimag.mat`
* [MASP](https://github.com/andresperezlopez/masp): a multichannel shoebox Room Impulse Response simulation library 

## Installation

`pip3 install -r reqs.txt`

## Multi-band RT60 values
The current state of the art datasets ([OpenSLR26](http://www.openslr.org/26/) and [OpenSLR28](http://www.openslr.org/28/)) are generated at 16kHz sampling rate and only consider a single RT60 value. However, in reality materials present different absorption coefficients for each frequency band. In the present work, on top of generating the IRs at a 48kHz sampling rate, we have analyzed the IEEE ICASSP2021 Acoustic Echo Cancellation Challenge Dataset, which provides plenty of multichannel RT60 values from real rooms. We have fitted these multi-band RT60 values as a set of six independent exponential distributions, one for each band.

The results of the fitting are shown below:

| band | alpha | beta |
|-----------------|-----------------|-----------------|
| 125Hz   | 1.72    | 0.39    |
| 250Hz   | 1.62    | 0.24    |
| 500Hz   | 1.93    | 0.14    |
| 1kHz    | 2.56    | 0.10    |
| 2kHz    | 4.12    | 0.09    |
| 4kHz    | 2.49    | 0.18    |

Sampling from these distributions approximates the real distributions.


## Source directivity
Current SOTA ([OpenSLR26](http://www.openslr.org/26/) and [OpenSLR28](http://www.openslr.org/28/)) also ignores the fact that the human speech is not omnidirectional. If wall absorption is frequency-dependant, we may also consider which speech reflections will be louder in a frequency-dependant manner. We have used an existing model of human speech directivity from [here](https://scholarsarchive.byu.edu/directivity/1/), that can be found in `directivity_parsing_matlab`. Once transformed to azimuth-elevation coordinates we can plot them in `results/speech_directivity.pdf` and we can use them in MASP. Note that for low frequencies speech is almost omnidirectional and that at high frequencies speech is highly directional towards the front and above the axis of propagation.

## Receiver directivity
In order to simulate the signals we would get on a headset or hearing aid devices, we have measured a set of HRTFs of a KU100 dummy head wearing a pair of Audifon Lewi R devices for the latter. We measured the HRTF using the sweep method with a single Genelec 8020 loudspeaker following a 50-point Lebedev grid. We cropped the impulse responses before the arrival of the first wall reflection and low frequencies were extended by the [LFE algorithm](https://zenodo.org/records/3928297). You may find these HRTFs in SOFA format at `sofa_files`.

The set of HRTFs has been used to build a 10-th order Ambisonics to binaural decoder following the Bilateral Magnitude Least Squares method [BiMagLS](https://github.com/isaacengel/BinauralSH). While our shoebox RIR simulator (MASP) outputs the Spherical Harmonics expansion of the IRs that already has the source directivity and multi-band features, the receiver directivity is applied when doing the Spherical Harmonics to binaural decoding. We provide this decoder in `decoders_ord10`.

## RIR dataset generation

We provide a Jupyter Notebook in which we have sampled the geometric and acoustic parameters of the 60.000 RIRs we have generated. Run `dataset_design.ipynb` to generate them again. We have already described how we sampled the RT60 values (from the six different exponential distributions), but the rest of parameters are sampled from uniform distributions whose limits are described in the following table:

| parameter | description | low | high |
|-----------------|-----------------|-----------------|-----------------|
| head_orient_azi | azimuth angle of the listening human head | -180º | 175º |
| head_orient_ele | elevation angle of the listening human head | -25º | 20º |
| distance  | distance from source to receiver | 0.5m | 3m |
| room_x | room length | 3m | 30m |
| room_y | room width | 0.5 * room_x | 1 * room_x |
| room_z | room height | 2.5m | 5m |
| head_x | receiver x coordinate | 0.35 * room_x | 0.65 * room_x |
| head_y | receiver y coordinate | 0.35 * room_y | 0.65 * room_y |
| head_z | receiver z coordinate  | 1m | 2m |
| src_orient_azi | azimuth of the speaking human* | -45º | 45º |
| src_orient_ele | elevation of the speaking human* | -20º | 20º |

<sup>*with respect to the axis between source and receiver 

Running all the cells from `dataset_desing.ipynb` generates a CSV file containing all the parameters, called `meta_was25.csv`. This file is then taken by the script `generate_data.py` to synthesize the RIRs. However, take into account that running this script required 320GB of RAM and 64 CPU cores for two weeks approximately. We rented an Amazon Web Service EC2 instance for that purpose. We don't recommend using the multiprocessing python library in this case. To speed things up, we recommend splitting the CSV file into smaller parts and to run several `generate_data.py` scripts in parallel at the same time, allowing the OS to kill any of the process in case some room is particularly memory hungry. In case you only need the computed RIRs, we can also provide the wav files (5GB approx.) in zenodo.

## Speech and Noise Datasets
We have used [DNS5](https://github.com/microsoft/DNS-Challenge) speech and noise data. Follow the instructions there to download it. Please be prepared for the following sizes once decompressed, and have at least twice the space available in your drive in order to generate HDF5 files.
| type | set | size |
|-----------------|-----------------|-----------------|
| speech | train | 344.4GB |
| speech | validation | 54.3GB |
| speech | test | 114.3GB |
| noise | train | 42.9GB |
| noise | validation | 9.2GB |
| noise | test |9.2GB |

URLs can be found in the [DNS5](https://github.com/microsoft/DNS-Challenge) repo's `download-dns-challenge-4.sh`script. Once downloaded and extracted, list all the individual files and split 70% for training, 15% for validation and 15% for test. We have performed a room and speaker-based separation, making sure that there is no contamination between sets. This means that if a speaker or a room appears in the training set, it won't appear in the validation or test sets. We provide these text files in `rir_generator/textfiles`.

Finally, for each text file, encapsulate all the tracks into big HDF5 files by doing:
```
cd DeepFilterNet

python df/scripts/prepare_data.py --sr 48000 rir <train_rir_textfile> TRAIN_SET_RIR

python df/scripts/prepare_data.py --sr 48000 speech <train_speech_textfile> TRAIN_SET_SPEECH

python df/scripts/prepare_data.py --sr 48000 noise <train_noise_textfile> TRAIN_SET_NOISE
```
Repeat for validation and test sets. The output paths for each HDF5 file should correspond with the ones in `DeepFilterNet/dconf.cfg`.

## Train configuration

Next, to train the DeepFilternet3 model with its default hyperparameters, the training script requires
* a model directory that contains a config files such as `trained_model/config.ini`
    * in this configuration file, we have modified `p_reverb`=1.0 to apply the HA RIRs to all the training utterances. The rest of hyperparameters are kept default.
    * in this folder, the checkpoints and summaries of the model will be saved
* the `dconf.cfg` file that lists the HDF5 files and allow for sampling factors.
* the actual data directory containing the nine HDF5 files we have described in `dconf.cfg`.

## Evaluation
Finally, to run the objective evaluation scripts, you should install the requirements from the [URGENT challenge](https://github.com/urgent-challenge/urgent2024_challenge/). Then adjust the paths to DNSMOS models and test set speech, noise and RIRS in 
`eval_headset.py`, `eval_speakerphone.py` and `eval_real.py` before running them.

These will generate `.csv`files with evaluation metrics for every test utterance. For averaging and plotting the results, you can use `notebooks/results_plotter.ipynb` and `ttest_plotter.ipynb`.

## LICENSE
MIT license.
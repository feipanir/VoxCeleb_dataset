'''
DataLoader for training

From 0 - 364003 to train_list_part1.txt
From 364003 - 728006 to train_list_part2.txt
From 728006 - 1092009 to train_list_part3.txt
Total lines: 1092009

'''

import glob, numpy, os, random, soundfile, torch
import argparse
from scipy import signal
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pdb

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, split, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        self.split_id = split
        # Load and configure augmentation files
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        # Load data & labels
        self.data_list  = []
        self.data_label = []
        lines = open(train_list).read().splitlines()
        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name     = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)
                
        if self.split_id == 1:
            start = 0 
            end = 364003
        elif self.split_id == 2:
            start = 364003
            end = 728006
        elif self.split_id == 3:
            start = 728006
            end = 1092009
        else:
            raise ValueError("split must be 1, 2, or 3")

        print("Using data from line {} to {} in train_list.txt".format(start, end))
        self.data_list = self.data_list[start:end]
        self.data_label = self.data_label[start:end]

    def __getitem__(self, index):
        # Read the utterance and randomly select the segment
        audio, sr = soundfile.read(self.data_list[index])		
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio],axis=0)
        # Data Augmentation
        
        augtype = random.randint(0,5)
        augtype = 3 # Disable augmentation for debugging


        if augtype == 0:   # Original
            audio = audio
        elif augtype == 1: # Reverberation
            audio = self.add_rev(audio)
        elif augtype == 2: # Babble
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3: # Music
            audio = self.add_noise(audio, 'music')
        elif augtype == 4: # Noise
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5: # Television noise
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')
        return (torch.FloatTensor(audio[0]), self.data_label[index], self.data_list[index], sr)

    def __len__(self):
        return len(self.data_list)

    def add_rev(self, audio):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float64),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames * 160 + 240
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio

if __name__ == "__main__":
    fix_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, required=True)
    args = parser.parse_args()

    out_folder = "processed_segments"
    out_dir = "./processed_segments"

    os.makedirs(out_dir, exist_ok=True)

    trainset = train_loader(train_list = './dbsource/VoxCeleb2/train_list.txt',
                            train_path = './dbsource/VoxCeleb2/train/wav', 
                            musan_path = './dbsource/Others/musan_split', 
                            rir_path   = './dbsource/Others/RIRS_NOISES/simulated_rirs', 
                            num_frames = 200,
                            split = args.split)
                            
    loader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    label_txt = os.path.join(out_dir, "label_part{}.txt".format(args.split))
    with open(label_txt, "w") as lf:
        for i, (audio, label, datapath, sr) in enumerate(tqdm(loader, total=len(trainset), desc="Exporting segments")):
            pdb.set_trace()
            wav = audio[0].numpy()
            target_path = datapath[0].replace('data08', out_folder)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            soundfile.write(target_path, wav, int(sr.item()), subtype='FLOAT')
            lf.write(f"{target_path} {int(label[0].item())}\n")

            # wav_read, sr_read = soundfile.read(target_path, dtype='float32', always_2d=False)
            # assert sr_read == int(sr.item())
            # assert wav_read.shape == wav.shape
            # print(np.array_equal(wav, wav_read))
            
            # pdb.set_trace()

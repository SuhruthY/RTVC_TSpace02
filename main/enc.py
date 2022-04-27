import warnings
warnings.filterwarnings("ignore")

import os
import random
import argparse
import numpy as np

from glob import glob

from utils import *

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq

## ----- GLOBAL PARAMS -----------------------------------
# CONSTANTS
SAMPLE_RATE = 16000
PAR_N_FRAMES = 160
MODEL_EM_SIZE = 256
MEL_N_CHANNEL = 40
MEL_WNDW_STEP = 10  

# VARS
WORKERS = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_DEVICE = torch.device("cpu")

# ARGUMENTS
parser = argparse.ArgumentParser()

parser.add_argument("-bdir", "--baseDir", default="../data/audio")
parser.add_argument("-mdir", "--modelDir", default="../data/model")
parser.add_argument("-ptdir", "--pretrainedDir", default="../data/pretrained_models")

parser.add_argument("-hs", "--hiddenSize", default=256, type=int)
parser.add_argument("-nl", "--nlayers", default=3, type=int)
parser.add_argument("-lr", "--learningRate", default=1e-4, type=float)

parser.add_argument("-spkr", "--speakersPerBatch", default=8, type=int)
parser.add_argument("-utter", "--utterancesPerSpeaker", default=5, type=int)


parser.add_argument("-e", "--epochs", default=20, type=int)
parser.add_argument("-v", "--version", default="v00")

parser.add_argument("-limt", "--limitTrain", default="30")
parser.add_argument("-limv", "--limitValid", default="20")

parser.add_argument("-se", "--saveEvery", default="100")
parser.add_argument("-sm", "--saveModel", action="store_true")

task = parser.add_mutually_exclusive_group()
task.add_argument('-train', action='store_true')
task.add_argument('-tune', action='store_true')
task.add_argument('-test', action='store_true')
task.add_argument('-ftest', action='store_true')

args = parser.parse_args()

# print(args)

## UTILS   
get_limit = lambda val, total: int(val) if val.isdigit() else float(val) * total

class RandomCycler:
    def __init__(self, source):
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count):
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
        return out

    def __next__(self):
        return self.sample(1)[0]

class Utterance:
    def __init__(self, frames_fpath, ):
        self.frames_fpath = frames_fpath
        
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        frames = self.get_frames()
        if frames.shape[0] == n_frames: start = 0
        else: start = np.random.randint(0, frames.shape[0] - n_frames)
        end = start + n_frames
        return frames[start:end], (start, end)

class Speaker:
    def __init__(self, root):
        self.root = root
        self.utterances = None
        self.utterance_cycler = None
        
    def load_utterances(self): 
        self.utterances = [Utterance(upath) for upath in glob(self.root + "/*")]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        if self.utterances is None: self.load_utterances()

        utterances = self.utterance_cycler.sample(count)

        return [(u,) + u.random_partial(n_frames) for u in utterances]

class SpeakerBatch:
    def __init__(self, speakers, utterances_per_speaker, n_frames):
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        self.data = np.array([frames for s in speakers for _, frames, _ in self.partials[s]])

class SpeakerVerificationDataset(Dataset):
    def __init__(self, dataset_root):
        self.root = dataset_root
        speaker_dirs = [f for f in glob(self.root + f"/*")]
        self.speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=WORKERS, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, n_frames=PAR_N_FRAMES) 

## MODEL
class SpeakerEncoder(nn.Module):
    def __init__(
        self, 
        hidden_size=args.hiddenSize,
        num_layers=args.nlayers,
        learning_rate = args.learningRate,
        ):
        super().__init__()
        self.loss_device = LOSS_DEVICE
        
        self.lstm = nn.LSTM(input_size=MEL_N_CHANNEL,
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True).to(DEVICE)
        self.linear = nn.Linear(in_features=hidden_size, 
                                out_features=MODEL_EM_SIZE).to(DEVICE)
        self.relu = torch.nn.ReLU().to(DEVICE)
        
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(self.loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(self.loss_device)

        self.loss_fn = nn.CrossEntropyLoss().to(self.loss_device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def do_gradient_ops(self):
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        return embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        
    
    def similarity_matrix(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        return sim_matrix * self.similarity_weight + self.similarity_bias
    
    def loss(self, embeds):
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer

## TRAINING UTILS
def sync(device: torch.device):
    if device.type == "cuda": torch.cuda.synchronize(device)

def run(run_id, train_dl, valid_dl, model, epoch=0):
    step = 0   
    state_fpath = f"{args.modelDir}/encoder_{run_id}.pt"
    if os.path.isfile(state_fpath) and args.saveModel:
        checkpoint = torch.load(state_fpath)
        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model_state"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state"])
        model.optimizer.param_groups[0]["lr"] = learning_rate_init   

    model.train()

    res = {
        "loss":0, 
        "eer":0,
        "val_loss": 0,
        "val_eer": 0,
    }

    i = 0
    bsize = get_limit(args.limitTrain, len(train_dl)) 
    cur_limit = bsize
    cur_epoch = ( '0' * (len(str(args.epochs))-len(str(epoch))) + str(epoch))
    loadbar(i, bsize, f"Epoch: [{cur_epoch}/{args.epochs}]  {i}/{bsize}", length=50)
    for idx, speaker_batch in enumerate(train_dl):
        if idx > cur_limit-1: break

        # Forward pass - Training
        inputs = torch.from_numpy(speaker_batch.data).to(DEVICE)
        sync(DEVICE)

        embeds = model(inputs)
        sync(DEVICE)

        embeds_loss = embeds.view((args.speakersPerBatch, args.utterancesPerSpeaker, -1)).to(LOSS_DEVICE)
        loss, eer = model.loss(embeds_loss)
        
        sync(LOSS_DEVICE)

        # Backward pass
        model.zero_grad()
        loss.backward()

        model.do_gradient_ops()
        model.optimizer.step()

        res["loss"] += loss.detach().item()
        res["eer"] += eer
        
        cur_batch = ( '0' * (len(str(bsize))-len(str(i+1))) + str(i+1))
        p = f"Epoch [{cur_epoch}/{args.epochs}] {cur_batch}/{bsize}"
        
        if i+1 == bsize:
            cur_limit = get_limit(args.limitValid, len(valid_dl))
            cur_batch = ( '0' * (len(str(bsize))-len(str(i))) + str(i))
            s = f"- loss:{res['loss']/bsize:0.4f} - eer:{res['eer']/bsize:0.4f}"
            loadbar(i-1, bsize, p, s, length=50)
            for idx, speaker_batch in enumerate(valid_dl):

                if idx > bsize-1: break

                # Forward pass - Validating
                inputs = torch.from_numpy(speaker_batch.data).to(DEVICE)
                sync(DEVICE)

                embeds = model(inputs)
                sync(DEVICE)

                embeds_loss = embeds.view((args.speakersPerBatch, args.utterancesPerSpeaker, -1)).to(LOSS_DEVICE)
                loss, eer = model.loss(embeds_loss)

                sync(LOSS_DEVICE)
                
                res["val_loss"] += loss
                res["val_eer"] += eer
                                
                e = f" - val_loss:{loss:0.4f} - val_eer:{eer:0.4f}"
                loadbar(i-1, bsize, p, s+e, length=50)
                  
            cur_batch = ( '0' * (len(str(bsize))-len(str(i+1))) + str(i+1))
            # p = f"Epoch [{cur_epoch}/{epochs}] {cur_batch}/{bsize}"
            e = f" - val_loss:{res['val_loss']/bsize:0.4f} - val_eer:{res['val_eer']/bsize:0.4f}"
            loadbar(i+1, bsize, p, s+e, length=50)
        else:
            s = f"- loss:{loss:0.4f} - eer:{eer:0.4f}"
            loadbar(i+1, bsize, p, s, length=50)
            
        i+=1

        # Overwrite the latest version of the model
        save_every = get_limit(args.saveEvery, len(train_dl))
        if save_every != 0 and step % save_every == 0 and args.saveModel:
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)

        step += 1

def train():
    train_dl = SpeakerVerificationDataLoader(
        SpeakerVerificationDataset(args.baseDir + "/train"),
        args.speakersPerBatch,
        args.utterancesPerSpeaker,
        num_workers=WORKERS
    )

    valid_dl = SpeakerVerificationDataLoader(
        SpeakerVerificationDataset(args.baseDir + "/val"),
        args.speakersPerBatch,
        args.utterancesPerSpeaker,
        num_workers=WORKERS
    )

    model=SpeakerEncoder()

    for epoch in range(args.epochs): run(args.version, train_dl, valid_dl, model, epoch)

## FUNCTIONAILTY TEST UTILS
def testutils():
    base_dir = args.baseDir + "/train"

    utter = Utterance(glob(base_dir + "/1272/*")[0])
    arr = utter.get_frames()
    print(arr.shape)
    arr, _ = utter.random_partial(PAR_N_FRAMES)
    print(arr.shape)

    spkr = Speaker(glob(base_dir + "/*")[0])
    _, arr, _ = spkr.random_partial(4, PAR_N_FRAMES)[0] 
    print(arr.shape)

    speakers = [Speaker(spath) for spath in glob(base_dir+"/*")][:3]
    batch = SpeakerBatch(speakers, 4, PAR_N_FRAMES)
    print(batch.data.shape)

    dl = SpeakerVerificationDataLoader(SpeakerVerificationDataset(base_dir), 4, 5)
    _, arr, _ = dl.dataset.__getitem__(1).random_partial(4, 160)[0]
    print(arr.shape)

## INFERENCE
def load_model(weights_fpath):
    model = SpeakerEncoder()
    checkpoint = torch.load(weights_fpath, DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded encoder encoder.pt - trained to step {checkpoint['step']}")

def is_loaded():
    return model is None

def embed_frames_batch(frames_batch):
    if not is_loaded(): load_model(args.pretrainedDir + "/encoder.pt")

    frames = torch.from_numpy(frames_batch).to(DEVICE)
    embed = model.forward(frames).detach().cpu().numpy()
    return embed

## To Implement - use original wav files
# def compute_partial_slices(
#     n_samples, 
#     partial_utterance_n_frames=PAR_N_FRAMES,
#     min_pad_coverage=0.75, overlap=0.5):

#     assert 0 <= overlap < 1
#     assert 0 < min_pad_coverage <= 1

#     samples_per_frame = int((SAMPLE_RATE * MEL_WNDW_STEP / 1000))
#     n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
#     frame_step = max(int(np.round(partial_utterance_n_frames * (1 - overlap))), 1)

#     # Compute the slices
#     wav_slices, mel_slices = [], []
#     steps = max(1, n_frames - partial_utterance_n_frames + frame_step + 1)
#     for i in range(0, steps, frame_step):
#         mel_range = np.array([i, i + partial_utterance_n_frames])
#         wav_range = mel_range * samples_per_frame
#         mel_slices.append(slice(*mel_range))
#         wav_slices.append(slice(*wav_range))

#     # Evaluate whether extra padding is warranted or not
#     last_wav_range = wav_slices[-1]
#     coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
#     if coverage < min_pad_coverage and len(mel_slices) > 1:
#         mel_slices = mel_slices[:-1]
#         wav_slices = wav_slices[:-1]

#     return wav_slices, mel_slices


# def embed_utterance(wav, using_partials=True, return_partials=False, **kwargs):
#     # Process the entire utterance if not using partials
#     if not using_partials:
#         # frames = audio.wav_to_mel_spectrogram(wav)
#         embed = embed_frames_batch(frames[None, ...])[0]
#         if return_partials:
#             return embed, None, None
#         return embed

#     # Compute where to split the utterance into partials and pad if necessary
#     wave_slices, mel_slices = compute_partial_slices(len(wav), **kwargs)
#     max_wave_length = wave_slices[-1].stop
#     if max_wave_length >= len(wav):
#         wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

#     # Split the utterance into partials
#     # frames = audio.wav_to_mel_spectrogram(wav)
#     frames_batch = np.array([frames[s] for s in mel_slices])
#     partial_embeds = embed_frames_batch(frames_batch)

#     # Compute the utterance embedding from the partial embeddings
#     raw_embed = np.mean(partial_embeds, axis=0)
#     embed = raw_embed / np.linalg.norm(raw_embed, 2)

#     if return_partials:
#         return embed, partial_embeds, wave_slices
#     return embed


if args.train: train()
if args.ftest: testutils()

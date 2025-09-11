import torch
import librosa
import numpy as np
import torch.nn.functional as F
from numpy.lib import stride_tricks
import os

@torch.no_grad()
def get_hubert(audio_file, args):
    aud_ori, sr = librosa.load(audio_file)
    audio_each_file = librosa.resample(aud_ori, orig_sr=sr, target_sr=args.audio_sr)
    from transformers import Wav2Vec2Processor, HubertModel
    print("Loading the Wav2Vec2 Processor...")
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("./facebook/hubert-large-ls960-ft")
    print("Loading the HuBERT Model...")
    hubert_model = HubertModel.from_pretrained("./facebook/hubert-large-ls960-ft")
    hubert_model.eval().cuda()
    if args.audio_rep == "onset+amplitude":
        frame_length = 1024
        shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
        strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
        rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
        amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
        audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=args.audio_sr, units='frames')
        onset_array = np.zeros(len(audio_each_file), dtype=float)
        onset_array[audio_onset_f] = 1.0
        mel = librosa.feature.melspectrogram(y=audio_each_file, sr=args.audio_sr, n_mels=128, hop_length=int(args.audio_sr/30))
        mel = mel[..., :-1]
        audio_emb = torch.from_numpy(np.swapaxes(mel, -1, -2))
        audio_emb = audio_emb.unsqueeze(0)

        hubert_feat, hubert_hid = get_hubert_from_16k_speech_long(hubert_model, wav2vec2_processor,
                                                                                        torch.from_numpy(aud_ori).unsqueeze(0),
                                                                                        )
        hubert_feat = F.interpolate(hubert_feat.swapaxes(-1, -2).unsqueeze(0),
                                                    size=audio_emb.shape[-2], mode='linear', align_corners=True).swapaxes(-1, -2)
        hubert_hid = F.interpolate(hubert_hid.swapaxes(-1, -2).unsqueeze(0),
                                                    size=audio_emb.shape[-2], mode='linear', align_corners=True).swapaxes(-1, -2)
        hubert_feat, hubert_hid = project_hubert(hubert_feat, hubert_hid)
        return hubert_feat, hubert_hid

@torch.no_grad()
def get_hubert_from_16k_speech_long( hubert_model, wav2vec2_processor, speech, device="cuda:0"):
    hubert_model = hubert_model.to(device)
    # if speech.ndim ==2:
    #     speech = speech[:, 0] # [T, 2] ==> [T,]
    input_values_all = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000).input_values.squeeze(0)  # [1, T]
    input_values_all = input_values_all.to(device)
    # For long audio sequence, due to the memory limitation, we cannot process them in one run
    # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
    # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
    # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
    # We have the equation to calculate out time step: T = floor((t-k)/s)
    # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
    # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
    kernel = 400
    stride = 320
    clip_length = stride * 1000
    num_iter = input_values_all.shape[1] // clip_length
    expected_T = (input_values_all.shape[1] - (kernel - stride)) // stride
    res_lst = []
    hid_list = []
    for i in range(num_iter):
        if i == 0:
            start_idx = 0
            end_idx = clip_length - stride + kernel
        else:
            start_idx = clip_length * i
            end_idx = start_idx + (clip_length - stride + kernel)
        input_values = input_values_all[:, start_idx: end_idx]
        hidden_states = hubert_model.forward(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        true_hidden = hubert_model.forward(input_values,output_hidden_states=True).hidden_states
        hid_list.append(true_hidden[11][0])
        res_lst.append(hidden_states[0])
    if num_iter > 0:
        input_values = input_values_all[:, clip_length * num_iter:]
    else:
        input_values = input_values_all
    # if input_values.shape[1] != 0:
    if input_values.shape[1] >= kernel:  # if the last batch is shorter than kernel_size, skip it
        hidden_states = hubert_model(input_values).last_hidden_state  # [B=1, T=pts//320, hid=1024]
        true_hidden = hubert_model.forward(input_values,output_hidden_states=True).hidden_states
        hid_list.append(true_hidden[11][0])
        res_lst.append(hidden_states[0])

    ret = torch.cat(res_lst, dim=0).cpu()  # [T, 1024]
    hid = torch.cat(hid_list, dim=0).cpu()
    # assert ret.shape[0] == expected_T
    assert abs(ret.shape[0] - expected_T) <= 1
    if ret.shape[0] < expected_T:
        ret = torch.nn.functional.pad(ret, (0, 0, 0, expected_T - ret.shape[0]))
    else:
        ret = ret[:expected_T]
    if hid.shape[0] < expected_T:
        hid = torch.nn.functional.pad(hid, (0, 0, 0, expected_T - hid.shape[0]))
    else:
        hid = hid[:expected_T]
    return ret, hid

def project_hubert(hubert_feat, hubert_hid):
    rop_matrix = _load_or_create_projection_matrix(
            rop_file="weights/rop_1024_256.pt", 
            in_dim=1024,
            out_dim=256,
            seed=42
        )
    rop_matrix = rop_matrix.to(hubert_feat.device)
    hubert_feat_proj = hubert_feat @ rop_matrix
    hubert_hid_proj = hubert_hid @ rop_matrix
    return hubert_feat_proj, hubert_hid_proj

def _load_or_create_projection_matrix(rop_file, in_dim=1024, out_dim=256, seed=42):
    """
    如果 rop_file 存在，则加载投影矩阵；
    如果不存在，则生成新的随机正交矩阵并保存到 rop_file。
    """
    if os.path.exists(rop_file):
        print(f"Loading ROP matrix from '{rop_file}'...")
        rop_matrix = torch.load(rop_file)
        return rop_matrix
    else:
        raise NotImplementedError("Please prepare the ROP matrix first.")

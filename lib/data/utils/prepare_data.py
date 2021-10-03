import os
import time

import h5py
import librosa
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def browse_folder(fd: str): # !![fd not enough explicit: change name]
    """ !![one-line summary] """
    paths, names = [], []
    for root, _, files in os.walk(fd):
        for name in files:
            if name.endswith('.wav'):
                filepath = os.path.join(root, name)
                names.append(name)
                paths.append(filepath)
    return names, paths

def float32_to_int16(x: tf.Tensor):
    """ !![one-line summary] """
    maxi = tf.reduce_max(tf.abs(x))
    if maxi > 1.0: # !![why?: comment]
        x /= maxi
    return tf.cast(x * (2.0**15 - 1), tf.int16)

def to_one_hot(k: int, classes_num: int):
    """ !![one-line summary] """
    return tf.one_hot(k, classes_num)

def pad_truncate_sequence(x: tf.Tensor, clip_samples: int):
    """ !![one-line summary, precise that x is an 1-D tensor] """
    if len(x) < clip_samples:
        return tf.concat([x, np.zeros(clip_samples - len(x))], axis=0)
    else:
        return x[0:clip_samples]

def get_target(audio_name: str, lb_to_idx: int):
    """ !![one-line summary] """
    return lb_to_idx[audio_name.split('.')[0]]

def int16_to_float32(x: tf.Tensor):
    """ !![one-line summary] """
    return tf.cast(x, tf.float32) / (2.0**15 - 1.0)

def traverse_folder(fd):
    paths = []
    names = []

    for root, dirs, files in os.walk(fd):
        for name in files:
            if name.endswith('.wav'):
                filepath = os.path.join(root, name)
                names.append(name)
                paths.append(filepath)

    return names, paths

def pack_audio_files_to_hdf5(path: str, packed_hdf5_path: str, 
                             clip_samples: int, classes_num: int, 
                             sample_rate: int, lb_to_idx: list, labels: list,
                             classes_size: dict):
    """ !![one-line summary] """
    # Define paths
    audios_dir = os.path.join(path)
    if not packed_hdf5_path.endswith('.h5'):
        packed_hdf5_path += '.h5'
    if os.path.exists(packed_hdf5_path):
        os.remove(packed_hdf5_path)
    if os.path.dirname(packed_hdf5_path) != '':
        os.makedirs(os.path.dirname(packed_hdf5_path), exist_ok=True)

    (audio_names, audio_paths) = traverse_folder(audios_dir)

    audio_names = sorted(audio_names)
    audio_paths = sorted(audio_paths)
    audios_num = len(audio_names)

    # targets are found using get_target
    targets = [get_target(audio_name, lb_to_idx) for audio_name in audio_names]

    meta_dict = {
        'audio_name': np.array(audio_names),
        'audio_path': np.array(audio_paths),
        'target': tf.convert_to_tensor(targets), # !![convert_to_tensor??]
        'fold': tf.convert_to_tensor(np.arange(len(audio_names)) % 10 + 1)} # !![convert_to_tensor??]

    feature_time = time.time()
    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name',
            shape=(audios_num,),
            dtype='S80')

        hf.create_dataset(
            name='waveform',
            shape=(audios_num, clip_samples),
            dtype=np.int16)

        hf.create_dataset(
            name='target',
            shape=(audios_num, classes_num),
            dtype=np.float32)

        hf.create_dataset(
            name='fold',
            shape=(audios_num,),
            dtype=np.int32)

        hf.attrs["classes_num"] = classes_num
        hf.attrs["clip_samples"] = clip_samples
        hf.attrs["sample_rate"] = sample_rate
        hf.attrs["labels"] = labels
        hf.attrs["classes_size"] = classes_size

        for i in tqdm(range(audios_num), total=audios_num):
            audio_name = meta_dict['audio_name'][i]
            fold = meta_dict['fold'][i]
            audio_path = meta_dict['audio_path'][i]
            target = meta_dict['target'][i]

            audio, _ = librosa.core.load(audio_path, sr=sample_rate,
                                            mono=True)
            audio = pad_truncate_sequence(audio, clip_samples)

            hf['audio_name'][i] = audio_name.encode() # !![why encode?]
            hf['waveform'][i] = float32_to_int16(audio)
            hf['target'][i] = to_one_hot(target, classes_num)
            hf['fold'][i] = fold

    print(f'Write hdf5 to {packed_hdf5_path}')
    print(f'Time: {time.time() - feature_time:.3f} sec')



import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold
from typing import List, Tuple, Dict

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse the ICBHI dataset filename to extract metadata.
    Format: [PatientID]_[RecordingIndex]_[ChestLocation]_[AcquisitionMode]_[Device].wav
    """
    base_name = os.path.basename(filename)
    if base_name.endswith('.wav'):
        base_name = base_name[:-4]
    
    parts = base_name.split('_')
    if len(parts) != 5:
        raise ValueError(f"Invalid filename format: {filename}")
        
    return {
        'patient_id': int(parts[0]),
        'recording_index': parts[1],
        'chest_location': parts[2],
        'acquisition_mode': parts[3],
        'device': parts[4]
    }

def get_device_from_filename(filename: str) -> str:
    """Extract device name from filename."""
    return parse_filename(filename)['device']

def load_annotations(annotation_path: str) -> List[Dict[str, float]]:
    """
    Parse the .txt annotation file.
    Format: start_time \t end_time \t crackle(0/1) \t wheeze(0/1)
    Class mapping: (0, 0) -> 0, (1, 0) -> 1, (0, 1) -> 2, (1, 1) -> 3
    """
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                continue
            start, end, crackle, wheeze = parts
            start, end = float(start), float(end)
            crackle, wheeze = int(float(crackle)), int(float(wheeze))
            
            label = 0
            if crackle == 1 and wheeze == 0:
                label = 1
            elif crackle == 0 and wheeze == 1:
                label = 2
            elif crackle == 1 and wheeze == 1:
                label = 3
                
            annotations.append({
                'start': start,
                'end': end,
                'crackle': crackle,
                'wheeze': wheeze,
                'label': label
            })
    return annotations

def create_patient_independent_splits(file_list: List[str], k_folds: int = 5, seed: int = 42) -> List[Tuple[List[str], List[str]]]:
    """
    Create k-fold splits where all files from a patient are in the same fold.
    """
    patient_ids = [parse_filename(f)['patient_id'] for f in file_list]
    
    gkf = GroupKFold(n_splits=k_folds)
    splits = []
    
    for train_idx, test_idx in gkf.split(file_list, groups=patient_ids):
        train_files = [file_list[i] for i in train_idx]
        test_files = [file_list[i] for i in test_idx]
        splits.append((train_files, test_files))
        
    return splits

class ICBHIDataset(Dataset):
    """ICBHI Dataset for PyTorch."""
    def __init__(self, audio_dir: str, annotation_dir: str, file_list: List[str], 
                 sample_rate: int = 16000, duration_sec: float = 5.0, transform=None):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.file_list = file_list
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.target_samples = int(self.sample_rate * self.duration_sec)
        self.transform = transform
        
        self.index_mapping = []
        for filename in self.file_list:
            base_name = os.path.basename(filename)
            if base_name.endswith('.wav'):
                base_name = base_name[:-4]
            ann_path = os.path.join(self.annotation_dir, f"{base_name}.txt")
            if os.path.exists(ann_path):
                anns = load_annotations(ann_path)
                for ann in anns:
                    self.index_mapping.append({
                        'filename': filename,
                        'annotation': ann
                    })

    def __len__(self) -> int:
        return len(self.index_mapping)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        item = self.index_mapping[index]
        filename = item['filename']
        ann = item['annotation']
        
        audio_path = os.path.join(self.audio_dir, filename)
        
        # Load full audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Slice to start and end
        start_sample = int(ann['start'] * self.sample_rate)
        end_sample = int(ann['end'] * self.sample_rate)
        
        sliced_waveform = waveform[:, start_sample:end_sample]
        
        # Pad or truncate
        if sliced_waveform.shape[1] > self.target_samples:
            # Truncate
            sliced_waveform = sliced_waveform[:, :self.target_samples]
        elif sliced_waveform.shape[1] < self.target_samples:
            # Pad
            padding = self.target_samples - sliced_waveform.shape[1]
            sliced_waveform = F.pad(sliced_waveform, (0, padding))
            
        if self.transform:
            sliced_waveform = self.transform(sliced_waveform)
            
        return sliced_waveform, ann['label']

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training."""
        counts = torch.zeros(4)
        for item in self.index_mapping:
            counts[item['annotation']['label']] += 1
            
        total = counts.sum()
        # To avoid division by zero
        counts = torch.clamp(counts, min=1.0)
        weights = total / (4.0 * counts)
        return weights

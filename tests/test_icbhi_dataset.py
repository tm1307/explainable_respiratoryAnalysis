import os
import pytest
import torch
import torchaudio
from src.data.icbhi_dataset import (
    parse_filename,
    load_annotations,
    create_patient_independent_splits,
    get_device_from_filename,
    ICBHIDataset
)

def test_parse_filename():
    filename = '101_1b1_Al_sc_Meditron.wav'
    parsed = parse_filename(filename)
    assert parsed == {
        'patient_id': 101,
        'recording_index': '1b1',
        'chest_location': 'Al',
        'acquisition_mode': 'sc',
        'device': 'Meditron'
    }

def test_parse_filename_different_devices():
    devices = ['AKGC417L', 'Litt3200', 'LittC2SE', 'Meditron']
    for dev in devices:
        filename = f'102_1b1_Al_sc_{dev}.wav'
        parsed = parse_filename(filename)
        assert parsed['device'] == dev
        assert parsed['patient_id'] == 102

def test_load_annotations(tmp_path):
    ann_file = tmp_path / "101_1b1_Al_sc_Meditron.txt"
    content = "0.0\t1.5\t0\t0\n" \
              "1.5\t3.0\t1\t0\n" \
              "3.0\t4.5\t0\t1\n" \
              "4.5\t6.0\t1\t1\n"
    ann_file.write_text(content)
    
    anns = load_annotations(str(ann_file))
    assert len(anns) == 4
    
    assert anns[0] == {'start': 0.0, 'end': 1.5, 'crackle': 0, 'wheeze': 0, 'label': 0}
    assert anns[1] == {'start': 1.5, 'end': 3.0, 'crackle': 1, 'wheeze': 0, 'label': 1}
    assert anns[2] == {'start': 3.0, 'end': 4.5, 'crackle': 0, 'wheeze': 1, 'label': 2}
    assert anns[3] == {'start': 4.5, 'end': 6.0, 'crackle': 1, 'wheeze': 1, 'label': 3}

def test_class_mapping():
    # Tested indirectly in test_load_annotations
    pass

def test_patient_independent_splits():
    file_list = []
    for pid in range(101, 106):  # 5 patients
        for idx in range(4):     # 4 files each
            file_list.append(f"{pid}_{idx}_Al_sc_Meditron.wav")
            
    splits = create_patient_independent_splits(file_list, k_folds=5, seed=42)
    assert len(splits) == 5
    
    for train_files, test_files in splits:
        train_pids = set([parse_filename(f)['patient_id'] for f in train_files])
        test_pids = set([parse_filename(f)['patient_id'] for f in test_files])
        assert len(train_pids.intersection(test_pids)) == 0

def test_patient_independent_splits_all_patients_covered():
    file_list = []
    for pid in range(101, 106):
        for idx in range(4):
            file_list.append(f"{pid}_{idx}_Al_sc_Meditron.wav")
            
    splits = create_patient_independent_splits(file_list, k_folds=5, seed=42)
    all_test_files = []
    for train_files, test_files in splits:
        all_test_files.extend(test_files)
        
    assert len(all_test_files) == len(file_list)
    assert set(all_test_files) == set(file_list)

def test_get_device_from_filename():
    assert get_device_from_filename('101_1b1_Al_sc_Meditron.wav') == 'Meditron'

def test_dataset_creation(tmp_path):
    audio_dir = tmp_path / "audio"
    ann_dir = tmp_path / "ann"
    audio_dir.mkdir()
    ann_dir.mkdir()
    
    file_list = []
    
    # Create 2 files
    for i, pid in enumerate([101, 102]):
        filename = f"{pid}_1b1_Al_sc_Meditron.wav"
        file_list.append(filename)
        
        # Audio: 10 seconds of random noise at 16000Hz
        waveform = torch.randn(1, 160000)
        torchaudio.save(str(audio_dir / filename), waveform, 16000)
        
        # Ann: 2 cycles per file
        ann_content = "0.0\t2.5\t0\t0\n2.5\t7.5\t1\t1\n"
        (ann_dir / f"{pid}_1b1_Al_sc_Meditron.txt").write_text(ann_content)
        
    dataset = ICBHIDataset(
        audio_dir=str(audio_dir),
        annotation_dir=str(ann_dir),
        file_list=file_list,
        sample_rate=16000,
        duration_sec=5.0
    )
    
    assert len(dataset) == 4
    
    for i in range(len(dataset)):
        waveform, label = dataset[i]
        assert waveform.shape == (1, 80000)  # 5.0s * 16000Hz
        assert label in [0, 3]

    weights = dataset.get_class_weights()
    assert weights.shape == (4,)

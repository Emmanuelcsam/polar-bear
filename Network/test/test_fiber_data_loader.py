#!/usr/bin/env python3
"""
Test suite for FiberOpticsDataLoader from fiber_data_loader.py
"""

import pytest
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock, call

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiber_data_loader import FiberOpticsDataset, FiberOpticsDataLoader, ReferenceDataLoader, StreamingDataLoader


class TestFiberOpticsDataset:
    """Test cases for FiberOpticsDataset class"""
    
    @pytest.fixture
    def mock_tensor_processor(self):
        """Create mock tensor processor"""
        mock_proc = Mock()
        mock_proc.load_tensor.return_value = torch.randn(3, 224, 224)
        mock_proc.tensorize_image.return_value = torch.randn(1, 3, 224, 224).squeeze(0)
        return mock_proc
    
    @pytest.fixture
    def sample_data_paths(self, tmp_path):
        """Create sample data paths with files"""
        # Create tensorized data
        tensor_dir = tmp_path / "tensorized"
        tensor_dir.mkdir()
        for i in range(5):
            torch.save(torch.randn(3, 224, 224), tensor_dir / f"tensor_{i}.pt")
        
        # Create image data
        image_dir = tmp_path / "images"
        image_dir.mkdir()
        for i in range(5):
            # Create dummy image file
            (image_dir / f"image_{i}.jpg").touch()
        
        return [tensor_dir, image_dir]
    
    @pytest.fixture
    @patch('fiber_data_loader.TensorProcessor')
    def dataset(self, mock_processor_class, mock_tensor_processor, sample_data_paths):
        """Create dataset instance"""
        mock_processor_class.return_value = mock_tensor_processor
        
        dataset = FiberOpticsDataset(
            data_paths=sample_data_paths,
            use_augmentation=False
        )
        return dataset
    
    def test_initialization(self, dataset, sample_data_paths):
        """Test dataset initialization"""
        assert dataset is not None
        assert len(dataset.data_files) == 10  # 5 tensors + 5 images
        assert hasattr(dataset, 'tensor_processor')
        assert hasattr(dataset, 'transform')
    
    def test_len(self, dataset):
        """Test dataset length"""
        assert len(dataset) == 10
    
    def test_getitem_tensor(self, dataset, mock_tensor_processor):
        """Test getting tensor item"""
        # Get item that should be a .pt file
        item = dataset[0]
        
        assert 'image' in item
        assert 'path' in item
        assert 'index' in item
        assert isinstance(item['image'], torch.Tensor)
        assert item['image'].shape == (3, 224, 224)
        mock_tensor_processor.load_tensor.assert_called()
    
    def test_getitem_image(self, dataset, mock_tensor_processor):
        """Test getting image item"""
        # Get item that should be a .jpg file
        item = dataset[5]  # Should be an image file
        
        assert 'image' in item
        assert 'path' in item
        assert isinstance(item['image'], torch.Tensor)
        mock_tensor_processor.tensorize_image.assert_called()
    
    def test_get_weights(self, dataset):
        """Test sample weight calculation"""
        weights = dataset.get_weights()
        
        assert len(weights) == len(dataset)
        assert all(w == 1.0 for w in weights)  # Default weights
    
    @patch('fiber_data_loader.TensorProcessor')
    def test_with_augmentation(self, mock_processor_class, mock_tensor_processor, sample_data_paths):
        """Test dataset with augmentation"""
        mock_processor_class.return_value = mock_tensor_processor
        
        dataset = FiberOpticsDataset(
            data_paths=sample_data_paths,
            use_augmentation=True,
            augmentation_prob=1.0  # Always augment for testing
        )
        
        # Mock augmentation
        with patch.object(dataset.tensor_processor, 'augment_tensor') as mock_augment:
            mock_augment.return_value = torch.randn(3, 224, 224)
            
            item = dataset[0]
            mock_augment.assert_called()
    
    def test_invalid_index(self, dataset):
        """Test invalid index access"""
        with pytest.raises(IndexError):
            dataset[100]
    
    def test_filter_extensions(self, dataset):
        """Test file extension filtering"""
        files = dataset._filter_extensions([Path("test.pt"), Path("test.jpg"), 
                                          Path("test.txt"), Path("test.png")])
        
        assert len(files) == 3  # .pt, .jpg, .png
        assert all(f.suffix in ['.pt', '.jpg', '.png'] for f in files)


class TestFiberOpticsDataLoader:
    """Test cases for FiberOpticsDataLoader class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.TENSORIZED_DATA_PATH = Path("/data/tensorized")
        mock_cfg.DATA_PATH = Path("/data")
        mock_cfg.BATCH_SIZE = 32
        mock_cfg.NUM_WORKERS = 4
        mock_cfg.PIN_MEMORY = True
        return mock_cfg
    
    @pytest.fixture
    @patch('fiber_data_loader.get_config')
    @patch('fiber_data_loader.get_logger')
    def data_loader(self, mock_get_logger, mock_get_config, mock_config):
        """Create data loader instance"""
        mock_get_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Mock path existence
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.iterdir', return_value=[]):
                loader = FiberOpticsDataLoader()
        
        return loader
    
    def test_initialization(self, data_loader):
        """Test data loader initialization"""
        assert data_loader is not None
        assert hasattr(data_loader, 'config')
        assert hasattr(data_loader, 'logger')
        assert hasattr(data_loader, 'data_paths')
    
    @patch('fiber_data_loader.FiberOpticsDataset')
    @patch('fiber_data_loader.random_split')
    def test_get_data_loaders(self, mock_split, mock_dataset_class, data_loader):
        """Test getting train/val data loaders"""
        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset_class.return_value = mock_dataset
        
        mock_train = Mock()
        mock_val = Mock()
        mock_split.return_value = [mock_train, mock_val]
        
        # Get loaders
        train_loader, val_loader = data_loader.get_data_loaders(
            batch_size=16,
            train_ratio=0.8
        )
        
        # Verify
        mock_split.assert_called_once()
        assert mock_split.call_args[0][1] == [80, 20]  # 80% train, 20% val
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
    
    @patch('fiber_data_loader.FiberOpticsDataset')
    def test_get_full_loader(self, mock_dataset_class, data_loader):
        """Test getting full dataset loader"""
        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset
        
        loader = data_loader.get_full_loader(batch_size=8, shuffle=True)
        
        assert isinstance(loader, DataLoader)
        mock_dataset_class.assert_called_with(
            data_paths=data_loader.data_paths,
            use_augmentation=False
        )
    
    @patch('fiber_data_loader.StreamingDataLoader')
    def test_get_streaming_loader(self, mock_streaming_class, data_loader):
        """Test getting streaming loader"""
        mock_streaming = Mock()
        mock_streaming_class.return_value = mock_streaming
        
        loader = data_loader.get_streaming_loader(batch_size=1)
        
        assert loader == mock_streaming
        mock_streaming_class.assert_called_with(
            data_paths=data_loader.data_paths,
            batch_size=1,
            buffer_size=30
        )
    
    @patch('fiber_data_loader.FiberOpticsDataset')
    @patch('fiber_data_loader.WeightedRandomSampler')
    def test_weighted_sampling(self, mock_sampler_class, mock_dataset_class, data_loader):
        """Test weighted sampling setup"""
        # Setup dataset mock
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_dataset.get_weights = Mock(return_value=[1.0] * 100)
        mock_dataset_class.return_value = mock_dataset
        
        # Setup sampler mock
        mock_sampler = Mock()
        mock_sampler_class.return_value = mock_sampler
        
        # Get loader with weighted sampling
        with patch('fiber_data_loader.random_split') as mock_split:
            mock_train = Mock()
            mock_train.dataset = mock_dataset
            mock_val = Mock()
            mock_split.return_value = [mock_train, mock_val]
            
            train_loader, val_loader = data_loader.get_data_loaders(
                use_weighted_sampling=True
            )
        
        # Verify sampler was created
        mock_sampler_class.assert_called()
        
    def test_add_data_path(self, data_loader):
        """Test adding data paths"""
        initial_count = len(data_loader.data_paths)
        
        with patch('pathlib.Path.exists', return_value=True):
            data_loader.add_data_path("/new/path")
        
        assert len(data_loader.data_paths) == initial_count + 1
        assert Path("/new/path") in data_loader.data_paths


class TestReferenceDataLoader:
    """Test cases for ReferenceDataLoader class"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        mock_cfg = Mock()
        mock_cfg.REFERENCE_PATH = Path("/data/reference")
        return mock_cfg
    
    @pytest.fixture
    def reference_files(self, tmp_path):
        """Create reference files"""
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        
        # Create reference tensors
        refs = {}
        for category in ['normal', 'defect', 'edge']:
            cat_dir = ref_dir / category
            cat_dir.mkdir()
            refs[category] = []
            
            for i in range(3):
                tensor = torch.randn(3, 224, 224)
                path = cat_dir / f"{category}_{i}.pt"
                torch.save(tensor, path)
                refs[category].append(path)
        
        return ref_dir, refs
    
    @pytest.fixture
    @patch('fiber_data_loader.get_config')
    @patch('fiber_data_loader.get_logger')
    def ref_loader(self, mock_get_logger, mock_get_config, mock_config, reference_files):
        """Create reference loader instance"""
        ref_dir, _ = reference_files
        mock_config.REFERENCE_PATH = ref_dir
        mock_get_config.return_value = mock_config
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        loader = ReferenceDataLoader()
        return loader
    
    def test_initialization(self, ref_loader):
        """Test reference loader initialization"""
        assert ref_loader is not None
        assert hasattr(ref_loader, 'reference_cache')
        assert len(ref_loader.reference_cache) > 0
    
    def test_load_references(self, ref_loader, reference_files):
        """Test loading reference tensors"""
        _, expected_refs = reference_files
        
        # Check all categories loaded
        for category in ['normal', 'defect', 'edge']:
            assert category in ref_loader.reference_cache
            assert len(ref_loader.reference_cache[category]) == 3
    
    def test_get_reference_tensor(self, ref_loader):
        """Test getting specific reference tensor"""
        tensor = ref_loader.get_reference_tensor('normal', 0)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
    
    def test_get_all_references(self, ref_loader):
        """Test getting all references for a category"""
        tensors = ref_loader.get_all_references('normal')
        
        assert isinstance(tensors, list)
        assert len(tensors) == 3
        assert all(isinstance(t, torch.Tensor) for t in tensors)
    
    def test_get_random_reference(self, ref_loader):
        """Test getting random reference"""
        tensor = ref_loader.get_random_reference('defect')
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
    
    def test_get_reference_statistics(self, ref_loader):
        """Test reference statistics"""
        stats = ref_loader.get_reference_statistics()
        
        assert 'total_references' in stats
        assert 'categories' in stats
        assert stats['total_references'] == 9  # 3 categories × 3 files
        assert len(stats['categories']) == 3
    
    def test_invalid_category(self, ref_loader):
        """Test invalid category access"""
        result = ref_loader.get_reference_tensor('invalid', 0)
        assert result is None
        
        refs = ref_loader.get_all_references('invalid')
        assert refs == []
    
    def test_add_reference(self, ref_loader, tmp_path):
        """Test adding new reference"""
        # Create new reference
        new_ref = torch.randn(3, 224, 224)
        ref_path = tmp_path / "new_ref.pt"
        torch.save(new_ref, ref_path)
        
        # Add reference
        initial_count = len(ref_loader.reference_cache.get('custom', []))
        ref_loader.add_reference('custom', str(ref_path))
        
        # Verify
        assert 'custom' in ref_loader.reference_cache
        assert len(ref_loader.reference_cache['custom']) == initial_count + 1


class TestStreamingDataLoader:
    """Test cases for StreamingDataLoader class"""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset"""
        mock_ds = Mock()
        mock_ds.__len__ = Mock(return_value=100)
        mock_ds.__getitem__ = Mock(side_effect=lambda i: {
            'image': torch.randn(3, 224, 224),
            'path': f'image_{i}.jpg',
            'index': i
        })
        return mock_ds
    
    @pytest.fixture
    @patch('fiber_data_loader.FiberOpticsDataset')
    def stream_loader(self, mock_dataset_class, mock_dataset):
        """Create streaming loader instance"""
        mock_dataset_class.return_value = mock_dataset
        
        loader = StreamingDataLoader(
            data_paths=[Path("/data")],
            batch_size=4,
            buffer_size=10
        )
        return loader
    
    def test_initialization(self, stream_loader):
        """Test streaming loader initialization"""
        assert stream_loader is not None
        assert stream_loader.batch_size == 4
        assert stream_loader.buffer_size == 10
        assert hasattr(stream_loader, 'current_index')
    
    def test_iter(self, stream_loader):
        """Test iteration"""
        iterator = iter(stream_loader)
        assert iterator == stream_loader
        assert stream_loader.current_index == 0
    
    def test_next(self, stream_loader, mock_dataset):
        """Test getting next batch"""
        batch = next(stream_loader)
        
        assert 'image' in batch
        assert batch['image'].shape == (4, 3, 224, 224)  # batch_size = 4
        assert stream_loader.current_index == 4
    
    def test_circular_iteration(self, stream_loader, mock_dataset):
        """Test circular iteration when reaching end"""
        # Set near end
        stream_loader.current_index = 98
        
        # Get batch that wraps around
        batch = next(stream_loader)
        
        assert batch['image'].shape == (4, 3, 224, 224)
        assert stream_loader.current_index == 2  # Wrapped around
    
    def test_stop_iteration(self, stream_loader):
        """Test manual stop"""
        stream_loader.stop()
        
        # Should raise StopIteration
        with pytest.raises(StopIteration):
            next(stream_loader)
    
    def test_buffer_refill(self, stream_loader):
        """Test buffer refilling"""
        # Deplete buffer
        for _ in range(5):  # More than buffer_size/batch_size
            next(stream_loader)
        
        # Buffer should have been refilled
        assert len(stream_loader.buffer) > 0
    
    def test_reset(self, stream_loader):
        """Test reset functionality"""
        # Advance loader
        next(stream_loader)
        next(stream_loader)
        
        # Reset
        stream_loader.reset()
        
        assert stream_loader.current_index == 0
        assert len(stream_loader.buffer) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
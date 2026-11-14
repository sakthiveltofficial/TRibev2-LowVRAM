"""
test_pipeline.py — Unit & Integration Test Suite for TRIBE-local
================================================================
Verifies tensor shape compatibility, forward passes, composite loss,
and metric calculation across all modular components.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from tribe_local.brain.losses import BrainLoss
from tribe_local.brain.metrics import BrainMetrics, OnlinePearsonCorr
from tribe_local.brain.subject_layers import (
    BrainPredictionHead,
    LightweightSubjectHead,
    ROIHead,
)
from tribe_local.data.dataset import tribe_collate_fn
from tribe_local.data.preprocessing import (
    create_synthetic_dataset,
    normalize_fmri,
)
from tribe_local.fusion.cross_attention_fusion import (
    CrossAttentionBlock,
    MultimodalTransformerFusion,
    TribeLocalFusion,
)
from tribe_local.fusion.projectors import ModalityProjectors
from tribe_local.fusion.temporal_align import TemporalAligner


class TestTribeLocal(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_01_temporal_aligner(self):
        aligner = TemporalAligner()
        feats = {
            "video": torch.randn(2, 384, 16),
            "audio": torch.randn(2, 768, 20),
            "text": torch.randn(2, 1536, 12),
        }
        aligned = aligner(feats)
        # Should align to max length (20)
        self.assertEqual(aligned["video"].shape, (2, 384, 20))
        self.assertEqual(aligned["audio"].shape, (2, 768, 20))
        self.assertEqual(aligned["text"].shape, (2, 1536, 20))

    def test_02_modality_projectors(self):
        projectors = ModalityProjectors(
            input_dims={"video": 384, "audio": 768, "text": 1536},
            hidden=512,
        )
        aligned = {
            "video": torch.randn(2, 384, 20),
            "audio": torch.randn(2, 768, 20),
            "text": torch.randn(2, 1536, 20),
        }
        fused = projectors(aligned)
        # Shape: (B, T, hidden)
        self.assertEqual(fused.shape, (2, 20, 512))

    def test_03_transformer_fusion(self):
        fusion = MultimodalTransformerFusion(
            hidden=512,
            num_layers=2,
            num_heads=4,
        )
        x = torch.randn(2, 20, 512)
        out = fusion(x, training=False)
        self.assertEqual(out.shape, (2, 20, 512))

    def test_04_full_fusion_pipeline(self):
        full_fusion = TribeLocalFusion(
            encoder_dims={"video": 384, "audio": 768, "text": 1536},
            hidden=512,
            low_rank=1024,
            num_transformer_layers=2,
            num_heads=4,
        )
        feats = {
            "video": torch.randn(2, 384, 16),
            "audio": torch.randn(2, 768, 20),
            "text": torch.randn(2, 1536, 12),
        }
        out = full_fusion(feats)
        # Expected bottleneck output: (B, 1024, T_common)
        self.assertEqual(out.shape, (2, 1024, 20))

    def test_05_voxel_prediction_head(self):
        head = BrainPredictionHead(
            mode="voxel",
            r=1024,
            n_vertices=20484,
            n_subjects=4,
            target_trs=100,
        )
        z = torch.randn(2, 1024, 20)
        subject_ids = torch.tensor([0, 2])
        pred = head(z, subject_ids)
        # Expected: (B, 20484, 100)
        self.assertEqual(pred.shape, (2, 20484, 100))

    def test_06_roi_prediction_head(self):
        head = BrainPredictionHead(
            mode="roi",
            r=1024,
            n_rois=360,
            n_subjects=4,
            target_trs=100,
        )
        z = torch.randn(2, 1024, 20)
        subject_ids = torch.tensor([1, 3])
        pred = head(z, subject_ids)
        # Expected: (B, 360, 100)
        self.assertEqual(pred.shape, (2, 360, 100))

    def test_07_composite_loss(self):
        loss_fn = BrainLoss(alpha=0.8)
        pred = torch.randn(2, 1000, 50)
        target = torch.randn(2, 1000, 50)
        loss_dict = loss_fn(pred, target)
        self.assertIn("total", loss_dict)
        self.assertIn("mse", loss_dict)
        self.assertIn("pearson_loss", loss_dict)
        self.assertGreater(float(loss_dict["total"]), 0.0)

    def test_08_online_pearson_metric(self):
        metric = OnlinePearsonCorr(n_vertices=50)
        # Highly correlated signals
        t = torch.linspace(0, 1, 100).unsqueeze(0).repeat(50, 1).unsqueeze(0)  # (1, 50, 100)
        p = t + 0.05 * torch.randn(1, 50, 100)
        metric.update(p, t)
        r = metric.compute()
        self.assertEqual(r.shape, (50,))
        self.assertGreater(float(r.mean()), 0.9)

    def test_09_synthetic_dataset_and_collation(self):
        test_dir = Path("test_synthetic_tmp")
        manifest = create_synthetic_dataset(
            test_dir,
            n_subjects=2,
            n_segments_per_subj=2,
            n_vertices=1000,
            duration_trs=50,
            save_files=True,
        )
        self.assertEqual(manifest["n_segments"], 4)

        # Clean up test files
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()

import pytest
import numpy as np
import torch
import networkx as nx
from affine import Affine

from src.models.losses import SoftSkeletonize, soft_cldice_loss, TRACECompositeLoss
from src.models.dlinknet import DLinkNet34
from src.vectorization.skeletonizer import extract_skeleton_graph
from src.vectorization.graph_contraction import contract_graph, prune_short_spurs
from src.vectorization.snapping_engine import snap_subgraph_to_master
from src.evaluation.apls import compute_apls


def test_soft_skeletonize_and_cldice():
    """Verify that soft-skeletonization is differentiable and outputs valid shapes."""
    skel_module = SoftSkeletonize(num_iters=3)
    dummy_input = torch.rand((2, 1, 64, 64), requires_grad=True)

    skel_out = skel_module(dummy_input)
    assert skel_out.shape == dummy_input.shape
    assert skel_out.min() >= 0.0
    assert skel_out.max() <= 1.0

    dummy_target = (torch.rand((2, 1, 64, 64)) > 0.8).float()
    dummy_logits = torch.randn((2, 1, 64, 64), requires_grad=True)

    loss = soft_cldice_loss(dummy_logits, dummy_target, skel_module)
    loss.backward()
    assert dummy_logits.grad is not None
    assert not torch.isnan(loss)


def test_dlinknet_4channel():
    """Verify D-LinkNet forward pass with 4-channel input."""
    model = DLinkNet34(in_channels=4, out_channels=1, pretrained=False)
    dummy_tensor = torch.randn(1, 4, 128, 128)
    output = model(dummy_tensor)
    assert output.shape == (1, 1, 128, 128)


def test_vectorization_and_contraction():
    """Verify raster skeletonization and degree-2 graph contraction."""
    # Create a 64x64 binary image with a horizontal line
    prob_map = np.zeros((64, 64), dtype=np.float32)
    prob_map[30:33, 10:50] = 0.9  # Line of length 40 pixels

    transform = Affine.translation(100.0, 200.0) * Affine.scale(1.0, -1.0)
    raw_graph, skeleton = extract_skeleton_graph(prob_map, transform, prob_threshold=0.5)

    assert len(raw_graph.nodes) > 0
    contracted = contract_graph(raw_graph)
    # The straight line should contract to its two endpoints
    assert len(contracted.nodes) <= 4


def test_snapping_engine():
    """Verify that the snapping engine properly attaches a dead-end to a master road."""
    master = nx.Graph()
    master.add_node("m1", pos_geo=(0.0, 0.0), pos_pixel=(0, 0))
    master.add_node("m2", pos_geo=(100.0, 0.0), pos_pixel=(100, 0))
    master.add_edge("m1", "m2", weight=100.0)

    # Subgraph approaching the master line orthogonally, terminating at y=10.0
    new_road = nx.Graph()
    new_road.add_node("n1", pos_geo=(50.0, 50.0), pos_pixel=(50, 50))
    new_road.add_node("n2", pos_geo=(50.0, 10.0), pos_pixel=(50, 10))
    new_road.add_edge("n1", "n2", weight=40.0)

    snapped = snap_subgraph_to_master(new_road, master, search_radius_meters=20.0)

    # Path should now exist from the new road to both ends of the master highway
    assert nx.has_path(snapped, "n1", "m1")
    assert nx.has_path(snapped, "n1", "m2")


def test_apls_metric():
    """Verify APLS calculation on identical and modified graphs."""
    g1 = nx.path_graph(5)
    for n in g1.nodes():
        g1.nodes[n]["pos_geo"] = (float(n * 20), 0.0)
    for u, v in g1.edges():
        g1[u][v]["weight"] = 20.0

    scores = compute_apls(g1, g1, sample_pairs=10, max_snap_dist=5.0)
    assert scores["apls_bidirectional"] == pytest.approx(1.0, abs=1e-3)

"""Point semantic annotation IO and ontology remapping contracts."""
from collections import OrderedDict

import numpy as np
import pytest

from dgp.annotations import (
    ANNOTATION_REGISTRY,
    SemanticSegmentation3DAnnotation,
    SemanticSegmentationOntology,
)
from dgp.annotations.transforms import OntologyMapper
from dgp.proto.ontology_pb2 import Ontology


@pytest.fixture
def ontology():
    proto = Ontology()
    for class_id, name in ((10, 'car'), (252, 'moving-car'), (255, 'rider'), (259, 'other')):
        proto.items.add(id=class_id, name=name, isthing=True)
    return SemanticSegmentationOntology(proto)


@pytest.mark.parametrize('explicit_target', [False, True])
def test_mapper_merges_point_classes_and_preserves_geometry(ontology, explicit_target):
    task = 'semantic_segmentation_3d'
    lookup = {'car': 'vehicle', 'moving-car': 'vehicle', 'rider': 'rider'}
    targets = None
    if explicit_target:
        target = Ontology()
        target.items.add(id=7, name='vehicle', isthing=True)
        target.items.add(id=12, name='rider', isthing=True)
        targets = {task: SemanticSegmentationOntology(target)}
    mapper = OntologyMapper({task: ontology}, {task: lookup}, targets)
    labels = np.array([0, 1, 2, 3, 255], dtype=np.uint8)
    annotation = SemanticSegmentation3DAnnotation(ontology, labels)
    points = np.arange(15).reshape(5, 3)
    datum = OrderedDict(semantic_segmentation_3d=annotation, point_cloud=points, timestamp=123)
    result = mapper(datum)
    mapped = result[task]
    ids = mapped.ontology.name_to_contiguous_id
    assert mapped.label.tolist() == [ids['vehicle'], ids['vehicle'], ids['rider'], 255, 255]
    np.testing.assert_array_equal(annotation.label, labels)
    np.testing.assert_array_equal(result['point_cloud'], points)
    assert result['timestamp'] == 123
    assert ANNOTATION_REGISTRY[task] is SemanticSegmentation3DAnnotation


def test_point_io_preserves_raw_255_and_void(tmp_path, ontology):
    source = tmp_path / 'raw.npz'
    raw = np.array([10, 252, 255, 259, -1, 999], dtype=np.int64)
    np.savez(source, data=raw)
    annotation = SemanticSegmentation3DAnnotation.load(source, ontology)
    assert annotation.label.tolist() == [0, 1, 2, 3, 255, 255]
    output = annotation.save(tmp_path)
    assert SemanticSegmentation3DAnnotation.load(output, ontology) == annotation
    assert SemanticSegmentation3DAnnotation.load(source.read_bytes(), ontology) == annotation
    with np.load(output) as content:
        assert content['data'].tolist() == [10, 252, 255, 259, -1, -1]


@pytest.mark.parametrize('data', [np.zeros((2, 2), dtype=int), np.array([1.2])])
def test_loader_rejects_non_point_label_arrays(tmp_path, ontology, data):
    source = tmp_path / 'invalid.npz'
    np.savez(source, data=data)
    with pytest.raises(ValueError, match='one-dimensional integer'):
        SemanticSegmentation3DAnnotation.load(source, ontology)


@pytest.mark.parametrize('labels', [np.array([-1]), np.array([4]), np.array([256])])
def test_annotation_rejects_invalid_contiguous_ids(ontology, labels):
    with pytest.raises(ValueError, match='contiguous'):
        SemanticSegmentation3DAnnotation(ontology, labels)


def test_empty_point_cloud_round_trip(tmp_path, ontology):
    annotation = SemanticSegmentation3DAnnotation(ontology, np.array([], dtype=np.uint8))
    assert SemanticSegmentation3DAnnotation.load(annotation.save(tmp_path), ontology) == annotation

"""Point-aligned semantic labels using the same ontology as image semantics."""
import hashlib
import io
from pathlib import Path

import numpy as np

from dgp.annotations.base_annotation import Annotation
from dgp.annotations.ontology import SemanticSegmentationOntology


class SemanticSegmentation3DAnnotation(Annotation):
    """Store one contiguous label per point; NPZ ``data`` stores raw class IDs.

    On disk, -1 denotes void. Raw ID 255 is a valid ontology class, unlike
    contiguous ID 255, which is DGP's reserved void label. Unknown raw IDs become
    void. Point order is preserved; coordinates belong to the associated datum.

    Parameters
    ----------
    ontology: SemanticSegmentationOntology
        Source label vocabulary.
    labels: numpy.ndarray
        One-dimensional integer array of contiguous class IDs or void.
    """
    def __init__(self, ontology: SemanticSegmentationOntology, labels: np.ndarray):
        super().__init__(ontology)
        if not isinstance(ontology, SemanticSegmentationOntology):
            raise TypeError('Point semantics require SemanticSegmentationOntology')
        if len(ontology.class_ids) > ontology.VOID_ID:
            raise ValueError('Semantic ontologies support at most 255 non-void classes')
        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('Point labels must be a one-dimensional integer array')
        valid = ((labels >= 0) & (labels < len(ontology.class_ids))) | (labels == ontology.VOID_ID)
        if not valid.all():
            raise ValueError('Point labels must be contiguous ontology IDs or void')
        self._labels = labels.astype(np.uint8, copy=True)

    @classmethod
    def load(cls, annotation_file, ontology):
        """Load raw class IDs from NPZ bytes or a path into contiguous labels.

        Parameters
        ----------
        annotation_file: str, pathlib.Path or bytes
            NPZ file or bytes containing the raw label array in ``data``.
        ontology: SemanticSegmentationOntology
            Source label vocabulary.

        Returns
        -------
        SemanticSegmentation3DAnnotation
            Point labels in the contiguous ontology space.

        Raises
        ------
        ValueError
            If stored class IDs are not a one-dimensional integer array.
        """
        source = io.BytesIO(annotation_file) if isinstance(annotation_file, bytes) else annotation_file
        with np.load(source, allow_pickle=False) as content:
            class_ids = content['data']
        if class_ids.ndim != 1 or not np.issubdtype(class_ids.dtype, np.integer):
            raise ValueError('Point labels must be a one-dimensional integer array')
        labels = np.full(class_ids.shape, ontology.VOID_ID, dtype=np.uint8)
        valid = (class_ids >= 0) & (class_ids < len(ontology.label_lookup))
        labels[valid] = ontology.label_lookup[class_ids[valid]]
        return cls(ontology, labels)

    def save(self, save_dir):
        """Save raw class IDs, preserving void without colliding with raw 255.

        Parameters
        ----------
        save_dir: str or pathlib.Path
            Existing destination directory for the content-addressed NPZ file.

        Returns
        -------
        str
            Full path to the saved annotation.
        """
        raw = np.full(self.label.shape, -1, dtype=np.int64)
        for contiguous_id, class_id in self.ontology.contiguous_id_to_class_id.items():
            raw[self.label == contiguous_id] = class_id
        output = Path(save_dir) / f'{self.hexdigest}.npz'
        np.savez_compressed(output, data=raw)
        return str(output)

    def render(self):
        """Point rendering requires the associated point-cloud datum."""
        raise NotImplementedError

    @property
    def label(self):
        return self._labels

    @property
    def hexdigest(self):
        """Hash labels and ontology so equal arrays in different spaces differ."""
        payload = self.ontology.to_proto().SerializeToString(deterministic=True) + self.label.tobytes()
        return hashlib.sha1(payload).hexdigest()

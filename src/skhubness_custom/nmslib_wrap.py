# -*- coding: utf-8 -*-
"""

This is from https://github.com/VarIr/scikit-hubness/blob/main/skhubness/neighbors/_nmslib.py 
with a few changes to handle different number of neighbours found for various points. 
Also logs minimum and maximum number of neighbours to NMSlibTransformer.log.


"""

# -*- coding: utf-8 -*-
# SPDX-License-Identifier: BSD-3-Clause
# Original work: https://scikit-learn.org/stable/auto_examples/neighbors/approximate_nearest_neighbors.html
# Author: Tom Dupre la Tour (original work)
#         Roman Feldbauer (adaptions for scikit-hubness)
# PEP 563: Postponed Evaluation of Annotations
from __future__ import annotations
import logging
from typing import Tuple, Union

import numpy as np
import re

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

import os
from src.util import naming

try:
    import nmslib
except ImportError:
    nmslib = None  # pragma: no cover

from tempfile import mkstemp, NamedTemporaryFile

__all__ = [
    "NMSlibTransformer",
]


def write_to_log(text: str, output_folder: str) -> None:
    log_path = os.path.join(output_folder, 'NMSlibTransformer.log')
    with open(log_path, 'a') as f:
        f.write(text + '\n')
        

# For removing the indexes again.
def delete_indexes_if_exist():
    index_dir_def = '/dev/shm'
    index_name = 'skhubness_.*'
    
    if os.path.exists(index_dir_def):
        index_files = [file 
                           for file in os.listdir(index_dir_def)
                           if re.match(index_name, file)]
        for file in index_files:
            logging.info(f'Removing index: {file}')
            os.remove(os.path.join(index_dir_def, file))
    elif os.getenv('APPDATA'):    
        index_dir_usr = os.getenv('APPDATA').replace('Roaming', '')
        index_dir_usr = os.path.join(index_dir_usr, 'Local/Temp')
        
        if os.path.exists(index_dir_usr):
            index_files = [file 
                               for file in os.listdir(index_dir_usr)
                               if re.match(index_name, file)]
            for file in index_files:
                logging.info(f'Removing index: {file}')
                os.remove(os.path.join(index_dir_usr, file))


def create_tempfile_preferably_in_dir(suffix=None, prefix=None, directory=None, persistent: bool = False, ):
    """ Create a temporary file with precedence for directory if possible, in TMP otherwise.
    For example, this is useful to try to save into /dev/shm.
    """
    temp_file = mkstemp if persistent else NamedTemporaryFile
    try:
        handle = temp_file(suffix=suffix, prefix=prefix, dir=directory)
        warn = False
    except FileNotFoundError:
        handle = temp_file(suffix=suffix, prefix=prefix, dir=None)
        warn = True

    # Extract the path (as string)
    try:
        path = handle.name
    except AttributeError:
        _, path = handle

    if warn:
        logging.warning(f"Could not create temp file in {directory}. "
                        f"Instead, the path is {path}.")
    return path


class NMSlibTransformer(BaseEstimator, TransformerMixin):
    """Approximate nearest neighbors retrieval with NMSLIB (non-metric space library).

    Compatible with sklearn's KNeighborsTransformer.
    NMSLIB is an approximate nearest neighbor library,
    that builds (hierarchically navigable) small-world graphs from
    a large number of different dissimilarity measure.

    Parameters
    ----------
    n_neighbors: int, default = 5
        Number of neighbors to retrieve
    metric: str, default = "euclidean"
        Distance metric, allowed are "cosine", "euclidean", and many others.
        See ``NMSlibTransformer.valid_metrics`` for the complete list.
        Note that "cosinesimil" refers to cosine distances in nmslib.
    p : float, optional
        Set p to define Lp space when using ``metric=="lp"``.
    alpha : float, optional
        Set alpha when using sqfd metrics
    method : str, default = "hnsw",
        (Approximate) nearest neighbor method to use. Allowed are "hnsw", "sw-graph",
        "vp-tree", "napp", "simple_invindx", "brute_force". Methods have individual
        parameters that may be tuned.
    efConstruction : float, optional
        Increasing the value of efConstruction improves the quality of a constructed graph
        and leads to higher accuracy of search, at the cost of longer indexing times.
        Relevant for methods "sw-graph" and "hnsw".
    ef, efSearch : float, optional
        Increasing the value of ef ("hnsw") or efSearch ("sw-graph") improves
        recall at the expense of longer retrieval time.
        The reasonable range of values for these parameters is 100-2000.
        Relevant for method "hnsw" and "sw-graph", respectively.
    M, NN : float, optional
        The recall values are also affected by parameters NN (for "sw-graph") and M ("hnsw").
        Increasing the values of these parameters (to a certain degree) leads to better
        recall and shorter retrieval times (at the expense of longer indexing time).
        For low and moderate recall values (e.g., 60-80%) increasing these parameters
        may lead to longer retrieval times.
        The reasonable range of values for these parameters is 5-100.
        Relevant for method "hnsw" and "sw-graph", respectively.
    delaunay_type : int, one of 0, 1, 2, 3, optional
        There is a trade-off between retrieval performance and indexing time related
        to the choice of the pruning heuristic (controlled by the parameter delaunay_type).
        Specifically, by default delaunay_type is equal to 2. This default is generally quite good.
        Relevant for method "hnsw".
    post_processing: int, optional
        Defines the amount (and type) of postprocessing applied to the constructed graph.
        Value 0 means no postprocessing. Additional options are 1 and 2 (2 means more postprocessing).
        More post processing means longer index creation, and higher retrieval accuracy.
        Relevant for method "hnsw".
    skip_optimized_index : int, default = 0
        There is a pesky design decision in NMSLIB that an index does not necessarily contain the data points,
        which are loaded separately. HNSW chooses to include data points into the index in several important cases,
        which include the dense spaces for the Euclidean and the cosine distance.
        These optimized indices are created automatically whenever possible.
        However, this behavior can be overriden by setting the parameter skip_optimized_index to 1.
        Relevant for method "hnsw".
    desiredRecall, bucketSize, tuneK, tuneR, tuneQty, minExp, maxExp
        Parameters relevant for method "vp-tree".
        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for details.
    numPivot, numPivotIndex, numPivotSearch, hashTrickDim
        Parameters relevant for method "napp".
        See https://github.com/nmslib/nmslib/blob/master/manual/methods.md for details.
    n_jobs: int, default = 1
        Number of parallel jobs
    mmap_dir: str, default = 'auto'
        Memory-map the index to the given directory. This is required to make the class pickleable.
        If None, keep everything in main memory (NON pickleable index),
        if mmap_dir is a string, it is interpreted as a directory to store the index into,
        if "auto", create a temp dir for the index, preferably in /dev/shm on Linux.
    verbose: int, default = 0
        Verbosity level. If verbose >= 2, show progress bar on indexing.

    Attributes
    ----------
    valid_metrics:
        List of valid distance metrics/measures
    """
    # https://github.com/nmslib/nmslib/tree/master/manual
    # Out-commented metrics are supported by NMSlib, but not yet here.
    # If you need them, file an issue with scikit-hubness at GitHub.
    valid_metrics = [
        # "bit_hamming",
        # "bit_jaccard",
        # "jaccard_sparse",
        "l1",
        "l1_sparse",
        "euclidean",
        "sqeuclidean",
        "l2",
        "l2_sparse",
        "linf",
        "linf_sparse",
        "lp",
        "lp_sparse",
        "angulardist",
        "angulardist_sparse",
        "angulardist_sparse_fast",
        "jsmetrslow",
        "jsmetrfast",
        "jsmetrfastapprox",
        # "leven",
        # "sqfdminusfunc",
        # "sqfdheuristicfunc",
        # "sqfdgaussianfunc",
        # "sdivslow",
        "jsdivfast",
        "jsdivfastapprox",
        "cosine",
        "cosinesimil",
        "cosinesimil_sparse",
        "cosinesimil_sparse_fast",
        # "normleven",
        "kldivfast",
        "kldivfastrq",
        "kldivgenslow",
        "kldivgenfast",
        "kldivgenfastrq",
        # "itakurasaitoslow",
        # "itakurasaitofast",
        # "itakurasaitofastrq",
        # "renyidiv_slow",
        # "renyidiv_fast",
        "negdotprod_sparse_fast",
    ]

    def __init__(self, output_folder: str,
                 n_neighbors=5, metric="euclidean",
                 p: float = None,
                 alpha: float = None,
                 method: str = "hnsw",
                 efConstruction: float = None,
                 ef: float = None,
                 efSearch: float = None,
                 M: float = None,
                 NN: float = None,
                 delaunay_type: int = None,
                 post_processing: int = None,
                 skip_optimized_index: int = None,
                 desiredRecall=None, bucketSize=None, tuneK=None, tuneR=None, tuneQty=None, minExp=None, maxExp=None,
                 numPivot=None, numPivotIndex=None, numPivotSearch=None, hashTrickDim=None,
                 n_jobs: int = 1,
                 mmap_dir: str = "auto",
                 verbose: int = 0,
                 ):

        if nmslib is None:  # pragma: no cover
            raise ImportError(
                "Please install the nmslib package before using NMSlibTransformer.\n"
                "pip install nmslib\n"
                "For best performance, install from sources:\n"
                "pip install --no-binary :all: nmslib",
            ) from None

        self.output_folder = output_folder
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.alpha = alpha
        self.method = method

        # HNSW and sw-graph parameters
        self.efConstruction = efConstruction
        self.ef = ef
        self.efSearch = efSearch
        self.M = M
        self.NN = NN
        self.delaunay_type = delaunay_type
        self.post_processing = post_processing
        self.skip_optimized_index = skip_optimized_index

        # vp-tree parameters
        self.desiredRecall = desiredRecall
        self.bucketSize = bucketSize
        self.tuneK = tuneK
        self.tuneR = tuneR
        self.tuneQty = tuneQty
        self.minExp = minExp
        self.maxExp = maxExp

        # napp parameters
        self.numPivot = numPivot
        self.numPivotIndex = numPivotIndex
        self.numPivotSearch = numPivotSearch
        self.hashTrickDim = hashTrickDim

        self.n_jobs = n_jobs
        self.mmap_dir = mmap_dir
        self.verbose = verbose

    def _construct_index_params_dict(self):
        if self.method in ["hnsw", "sw-graph"]:
            index_params = {
                "efConstruction": self.efConstruction,
                "delaunay_type": self.delaunay_type,
                "post": self.post_processing,
                "skip_optimized_index": self.skip_optimized_index,
                "indexThreadQty": self.n_jobs,

            }
            if self.method == "hnsw":
                index_params["ef"] = self.ef
                index_params["M"] = self.M
            else:
                index_params["efSearch"] = self.efSearch
                index_params["NN"] = self.NN
        elif self.method == "vp-tree":
            index_params = {
                "desiredRecall": self.desiredRecall,
                "bucketSize": self.bucketSize,
                "tuneK": self.tuneK,
                "tuneR": self.tuneR,
                "tuneQty": self.tuneQty,
                "minExp": self.minExp,
                "maxExp": self.maxExp,
            }
        elif self.method == "napp":
            index_params = {
                "numPivot": self.numPivot,
                "numPivotIndex": self.numPivotIndex,
                "numPivotSearch": self.numPivotSearch,
                "hashTrickDim": self.hashTrickDim,
            }
        elif self.method in ["simple_invindx", "brute_force"]:
            index_params = {}
        else:
            raise ValueError(f'Unknown method: {self.method}. Use one of: '
                             f'"hnsw", "sw-graph", "vp-tree", "napp", "simple_invindx", "brute_force"')
        # We only pass parameters that are explicitly provided by the user
        index_params = {k: v for k, v in index_params.items() if v is not None}
        return index_params

    def _possibly_mmap_index(self):
        # TODO create a MemMapMixin and move this code there
        if isinstance(self.mmap_dir, str):
            directory = "/dev/shm" if self.mmap_dir == "auto" else self.mmap_dir
            self.neighbor_index_ = create_tempfile_preferably_in_dir(
                prefix="skhubness_",
                suffix=".nmslib",
                directory=directory,
            )
            if self.mmap_dir == "auto":
                logging.warning(
                    f"The index will be stored in {self.neighbor_index_}. "
                    f"It will NOT be deleted automatically, when this instance is destructed.",
                )
        else:  # e.g. None
            self.mmap_dir = None

    def fit(self, X, y=None) -> NMSlibTransformer:
        """ Build the NMSLIB index and insert data from X.

        Parameters
        ----------
        X: array-like
            Data to be indexed
        y: ignored

        Returns
        -------
        self: NMSlibTransformer
            An instance of NMSlibTransformer with a built index
        """
        space = {
            **{x: x for x in NMSlibTransformer.valid_metrics},
            "euclidean": "l2",
            "sqeuclidean": "l2",
            "cosine": "cosinesimil",
        }.get(self.metric, None)
        if space is None:
            raise ValueError(f"Invalid metric: {self.metric}")
        self.space_ = space
        self.sparse_ = "_sparse" in self.space_
        
        X: Union[np.ndarray, csr_matrix] = check_array(X, accept_sparse=self.sparse_)  # noqa
        n_samples, n_features = X.shape
        self.n_samples_in_ = n_samples
        self.n_features_in_ = n_features

        # Different nearest neighbor methods in NMSLIB have different parameters to tune,
        # and are passed as a dict to nmslib.init()
        self.index_params_ = self._construct_index_params_dict()

        # Save an index to disk or keep in memory, depending on self.mmap
        self._possibly_mmap_index()

        data_type = nmslib.DataType.DENSE_VECTOR
        dist_type = nmslib.DistType.FLOAT
        if "_sparse" in self.space_:
            data_type = nmslib.DataType.SPARSE_VECTOR

        self.data_type_ = data_type
        self.dist_type_ = dist_type
        
        # Some metrics require additional parameters
        space_params = {}
        if self.metric in ["lp", "lp_sparse"]:
            space_params["p"] = self.p
        self.space_params_ = space_params

        index: nmslib.dist.FloatIndex = nmslib.init(
            method=self.method,
            space=space,
            space_params=self.space_params_,
            data_type=self.data_type_,
            dtype=self.dist_type_,
        )

        index.addDataPointBatch(X)
        index.createIndex(
            index_params=self.index_params_,
            print_progress=(self.verbose >= 2),
        )

        if self.mmap_dir is None:
            self.neighbor_index_ = index
        else:
            index.saveIndex(self.neighbor_index_, save_data=True)

        # nmslib MAY return squared distances (https://github.com/nmslib/nmslib/issues/504#issuecomment-949710037)
        # Let's do a little heuristic, if this is the case here, and sqrt if necessary.
        self._do_sqrt = False
        if self.metric == "euclidean" and self.n_samples_in_ > 1:
            ((_, ind), (_, dist)) = index.knnQueryBatch(X[0:1, :], k=2)[0]
            sq_dist = np.sum(np.power(X[0, :] - X[ind, :], 2))
            if np.isclose(dist, sq_dist):
                self._do_sqrt = True

        return self

    def transform(self, X) -> csr_matrix:
        """ Create k-neighbors graph for the query objects in X.

        Parameters
        ----------
        X : array-like
            Query objects

        Returns
        -------
        kneighbors_graph : csr_matrix
            The retrieved approximate nearest neighbors in the index for each query.
        """
        check_is_fitted(self, "neighbor_index_")
        X: Union[np.ndarray, csr_matrix] = check_array(X, accept_sparse=self.sparse_)  # noqa

        n_samples_transform, n_features_transform = X.shape
        if n_features_transform != self.n_features_in_:
            raise ValueError(f"Shape of X ({n_features_transform} features) does not match "
                             f"shape of fitted data ({self.n_features_in_} features.")

        # Load memory-mapped nmslib.Index, unless it's already in main memory
        if isinstance(self.neighbor_index_, str):
            neighbor_index = nmslib.init(
                space=self.space_,
                space_params=self.space_params_,
                method=self.method,
                data_type=self.data_type_,
                dtype=self.dist_type_,
            )
            neighbor_index.loadIndex(self.neighbor_index_, load_data=True)
        else:
            neighbor_index = self.neighbor_index_

        # Do we ever need one additional neighbor (e.g., for self distances?)
        n_neighbors = self.n_neighbors + 1

        results = neighbor_index.knnQueryBatch(
            X,
            k=n_neighbors,
            num_threads=self.n_jobs,
        )
                
        indices, distances = zip(*results)
        
        lengths = list(map(len, indices))
        min_length = min(lengths)
        max_length = max(lengths)
        logging.info(f'Min length: {min_length}, max length: {max_length}')
        write_to_log(f'Min number of neighbours: {min_length}, max number of neighbours: {max_length}',
                     self.output_folder)
        write_to_log(' ',
                     self.output_folder)
        
        if min_length != max_length:
            indices = [np.array(i)[:min_length]
                       for i in indices]
            distances = [np.array(i)[:min_length]
                         for i in distances]    
            
        
        
        indices, distances = np.vstack(indices), np.vstack(distances)
        distances = distances.astype(X.dtype)
        
        # nmslib MAY return squared distances (https://github.com/nmslib/nmslib/issues/504#issuecomment-949710037)
        # Let's do a little heuristic, if this is the case here, and sqrt if necessary.
        if self._do_sqrt:
            try:
                np.sqrt(distances, out=distances)
            except TypeError:  # int types
                dtype = distances.dtype
                distances = np.sqrt(distances)
                np.round(distances, decimals=0, out=distances)
                distances = distances.astype(dtype)

        indptr = np.arange(
            start=0,
            stop=n_samples_transform * min_length + 1,
            step=min_length,
        )
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_in_),
        )

        return kneighbors_graph





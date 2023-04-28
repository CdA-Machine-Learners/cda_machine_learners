'''

A simple "Vector Database" using numpy to keep a matrix of vectors, and a
dictionary to keep metadata.

'''

import numpy as np
from typing import Dict, List, Any, Tuple

def cosine_similarity(vector, matrix):
    '''
    Compare a `vector` against each row of a `matrix` via cosine_similarity.

    Args:
      vector: np.array[dim]
      matrix: np.array[n, dim]

    Returns:
      np.array[n]
    '''
    # Normalize the input vector
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        raise ValueError("Input vector has zero magnitude.")
    normalized_vector = vector / vector_norm

    # Normalize each row in the input matrix
    matrix_norms = np.linalg.norm(matrix, axis=1).reshape(-1, 1)
    zero_norm_rows = np.isclose(matrix_norms, 0)
    if np.any(zero_norm_rows):
        raise ValueError("One or more rows in the input matrix have zero magnitude.")
    normalized_matrix = matrix / matrix_norms

    # Calculate the cosine similarity between the vector and each row in the matrix
    cosine_similarities = np.dot(normalized_matrix, normalized_vector)

    return cosine_similarities


# @@@@@@@@@@
# Tests

v = np.random.randn(10000) # a test vector
m = np.random.randn(21, 10000) # uncorrelated rows of vectors

# Add the test vector into each row.
#   * The lowest index row should be anticorrelated, when alpha is (-1)
#   * The middle row should not be correlated, when alpha is 0
#   * The final row should be correlated, when alpha is 1
for i, alpha in enumerate(np.linspace(-1, 1, 21)):
    m[i] = v * alpha + m[i] * (1 - np.abs(alpha))

# Find the  similarity of the  test vector `v`  to each row  of `m` that  it was
# gradually interpolated into.
sim = cosine_similarity(v, m)

# Assert monotonic increasing
for i in range(1, 21):
    assert sim[i-1] < sim[i]

# 0th index is anticorrelated
assert np.abs(sim[0]  - (-1)) < 0.1

# Middle index is not correlated
assert np.abs(sim[10] - 0   ) < 0.1

# Final index is perfectly correlated
assert np.abs(sim[20] - 1   ) < 0.1

# @@@@@@@@@@


class VectorDB:
    '''
    A simple "Vector DB" Implementation
    '''
    def __init__(self, embedding_len):
        self.embedding_len = embedding_len
        self.db = {}
        self.np_index = None # vectors into db
        self.np_embedding = None

    def upsert_multiple(self, xs: List[Tuple[int, np.ndarray, Any]]) -> None:
        '''Upsert a list of tuples: [(integer id, 1d vector, anything)].

        Args:
          xs: [(id, embedding, metadata)]

        '''
        for i, e, m in xs:
            self.db[i] = (e, m)
        self.rebuild_index()

    def rebuild_index(self) -> None:
        '''Rebuild `self.np_embedding`, and the indexes that correspond to each row of
        those embeddings, `self.np_index`.

        '''
        ix_embs = [(i, x[0]) for i, x in self.db.items()] # trim off metadata, keep db-ixs
        self.np_index = np.array([x[0] for x in ix_embs])
        self.np_embedding = np.array([x[1] for x in ix_embs])

    def query(self, embedding: np.ndarray, top_k : int):
        '''Return the `top_k` matches similar to your query `embedding`. Results are a
        list of dictionaries, with keys {score, index, embedding, metadata}.

        '''
        if self.np_embedding is None:
            return []

        sim = cosine_similarity(embedding, self.np_embedding)
        ixs = list(reversed(np.argsort(sim)))[:top_k]
        out = []
        for i in ixs:
            db_i = self.np_index[i]
            out.append({
                'score': sim[i],
                'index': db_i,
                'embedding': self.db[db_i][0],
                'metadata': self.db[db_i][1],
            })
        return out

# @@@@@@@@@@
# Tests

dim = 10000
index = VectorDB(dim)
m = np.random.randn(10, dim)
top_k = 5
top_index = 0

# 5 random vectors
test_vectors = [np.random.randn(dim) for _ in range (10)]
for v in test_vectors:
    # insert noised versions of test vectors into db
    for i, alpha in enumerate(np.linspace(0.1, 1, top_k)):
        emb = v * alpha + np.random.randn(dim) * (1 - alpha) # noised version
        index.upsert_multiple([(top_index, emb, f'ix:{top_index}')])
        top_index += 1

# Query each test vector, its nearest neighbors should be the noised versions of
# itself
for i, v in enumerate(test_vectors):
    res = index.query(v, top_k=top_k)
    # assert test_vector=0 is close to ixs 0-5
    # assert test_vector=1 is close to ixs 6-10, etc.
    for j, r in enumerate(reversed(res)):
        assert r['index'] == i*top_k + j

print('done')

# @@@@@@@@@@

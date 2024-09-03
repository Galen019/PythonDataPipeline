import numpy as np
from scipy.sparse._matrix import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents: list[str] = [
    "The cat in the hat disabled",
    "A cat is a fine pet ponies.",
    "Dogs and cats make good pets.",
    "I haven't got a hat.",
]

# Compute TF-IDF
vectorizer = TfidfVectorizer()
X: spmatrix = vectorizer.fit_transform(documents)

# Perform SVD using NumPy
U, S, Vt = np.linalg.svd(X.toarray(), full_matrices=False)  # type: ignore


# Pretty print matrices
def pretty_print_matrix(matrix, name):
    print(f"{name} matrix:")
    print(np.array2string(matrix, formatter={"float_kind": lambda x: "%.2f" % x}))


# Show results
pretty_print_matrix(U, "U")
print("Singular values:")
print(np.array2string(S, formatter={"float_kind": lambda x: "%.2f" % x}))
pretty_print_matrix(Vt, "Vt")

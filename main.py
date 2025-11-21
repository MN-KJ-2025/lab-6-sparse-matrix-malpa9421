# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if (not (isinstance(A, np.ndarray) or isinstance(A, sp.sparse.csc_array))):
        return None
    if A.ndim != 2:
        return None
    m, n = A.shape
    if m != n:
        return None

    bA = np.absolute(A)
    dA = bA.diagonal() #tak musi yc zeby działało zarówno dla zwykłego i żadkiego aray
    sums = np.sum(bA - np.diag(dA),1)
    #print(sums, dA)
    if np.all(sums < dA):
        return True
    return False

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not (isinstance(A,np.ndarray) and isinstance(x,np.ndarray) and isinstance(b,np.ndarray)):
        return None

    if len(x.shape) > 1 or len(b.shape) > 1: #można normalnie użyć tych danych, ale testy oczekuja none
        return None

    try: r = A @ x - b
    except Exception:
        return None
    

    return np.linalg.norm(r)

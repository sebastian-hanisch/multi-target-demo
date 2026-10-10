"""Gemeinsame Fixtures: Mini-Beispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def cost_example():
    """Zwei Spuren, zwei Messungen. Spur 0: (1; 5), Spur 1: (1,5; 4).

    Nächste Messung: beide Spuren nehmen Messung 0 (Kosten 1 und 1,5, zusammen 2,5).
    Globale Zuordnung: Spur 0 -> Messung 0 (1) und Spur 1 -> Messung 1 (4) = 5 gegen Spur 0 -> 1 (5) und Spur 1 -> 0 (1,5) = 6,5: wählt (0, 1).
    """
    return np.array([[1.0, 5.0], [1.5, 4.0]])

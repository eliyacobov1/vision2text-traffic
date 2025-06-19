import numpy as np


def lucas_kanade_flow(prev: np.ndarray, curr: np.ndarray, points: list[np.ndarray], window: int = 5) -> list[np.ndarray]:
    """Compute optical flow vectors using the Lucas-Kanade method.

    Args:
        prev: Grayscale previous frame.
        curr: Grayscale current frame.
        points: List of [x, y] coordinates to track.
        window: Half window size around each point.

    Returns:
        List of [vx, vy] flow vectors matching ``points``.
    """
    prev = prev.astype(float)
    curr = curr.astype(float)
    Ix = np.zeros_like(prev)
    Iy = np.zeros_like(prev)
    Iy[:, :-1] = prev[:, 1:] - prev[:, :-1]
    Ix[:-1, :] = prev[1:, :] - prev[:-1, :]
    It = curr - prev

    flows = []
    h, w = prev.shape
    for pt in points:
        x, y = int(round(pt[0])), int(round(pt[1]))
        x1, x2 = max(0, x - window), min(w - 1, x + window)
        y1, y2 = max(0, y - window), min(h - 1, y + window)
        Ix_patch = Ix[y1:y2 + 1, x1:x2 + 1].flatten()
        Iy_patch = Iy[y1:y2 + 1, x1:x2 + 1].flatten()
        It_patch = It[y1:y2 + 1, x1:x2 + 1].flatten()
        A = np.stack([Ix_patch, Iy_patch], axis=1)
        ATA = A.T @ A
        if np.linalg.det(ATA) < 1e-6:
            flows.append(np.zeros(2))
            continue
        nu = np.linalg.inv(ATA) @ A.T @ (-It_patch)
        flows.append(nu)
    return flows

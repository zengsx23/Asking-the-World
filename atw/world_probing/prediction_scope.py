"""Explicit future-answer scope for rigid prediction only."""
import math

PREDICTION_GUIDANCE = (
    'Predict what happens strictly AFTER the observed video cutoff. '
    'An event already seen before or at the cutoff is not an answer merely because it happened earlier. '
    'Historical proximity or contact does not establish a future collision. '
    'A collision not yet observed is not evidence that it will not happen. '
    'Use recent relative motion, direction and geometry; distance alone does not determine a collision. '
    'Evaluate every option as a future event. For next-event wording, compare qualifying future events, '
    'never rank historical contact times. Missing or stale observations are not evidence of absence.')


def applies(mode, engine):
    return mode == 'predictive' and engine == 'rigid'


def video_cutoff(video):
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count < 1:
            raise ValueError('Cannot determine the observed video cutoff')
        return (count - 1) / fps
    finally:
        capture.release()

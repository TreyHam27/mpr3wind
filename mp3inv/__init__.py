"""mp3inv: find an MP3 preimage of decoded PCM under a fixed reference decoder (minimp3)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
MINIMP3_H = os.path.join(ROOT, "third_party", "minimp3", "minimp3.h")

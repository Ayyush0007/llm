"""
microphone_sensor.py — Audio detection for Indian road robot.

Critical for India because:
  • Car horns are used as primary communication (not courtesy beeps)
  • Emergency vehicles (ambulance, police) use sirens
  • Construction / jackhammer nearby = narrow road ahead
  • Detecting honks from behind = someone wants to overtake

Detects:
  • Car horn (250–500 Hz impulse)
  • Ambulance/police siren (wailing 500–2000 Hz sweep)
  • Loud bang / collision
  • Crowd noise (pedestrian crossing / market area)

PHASE 1: Uses librosa on pre-recorded audio clips for CI testing
PHASE 2: USB microphone → PyAudio real-time stream on Jetson

Install:  pip install pyaudio librosa sounddevice numpy
Hardware: USB mic or 3.5mm mic → Jetson audio input
"""

import time
import threading
import numpy as np
import queue
from dataclasses import dataclass
from typing import Optional

# Sound event types
EVENT_HORN      = "horn"
EVENT_SIREN     = "siren"
EVENT_BANG      = "bang"
EVENT_CROWD     = "crowd"
EVENT_QUIET     = "quiet"


@dataclass
class AudioEvent:
    """Detected sound event."""
    event_type:  str   = EVENT_QUIET
    confidence:  float = 0.0
    db_level:    float = 0.0   # loudness in dBFS
    timestamp:   float = 0.0

    # Action hints
    should_stop:  bool = False   # emergency vehicle
    should_yield: bool = False   # horn from behind
    should_slow:  bool = False   # crowd detected ahead


class MicrophoneSensor:
    """
    Real-time audio event detector.
    Uses frequency analysis to classify road sounds.
    """

    SAMPLE_RATE  = 22050   # Hz
    CHUNK_SIZE   = 2048    # samples per analysis frame (~93ms)
    HORN_BAND    = (250,   500)   # Hz
    SIREN_BAND   = (500,  2000)   # Hz
    BANG_DB      = -10     # dBFS threshold for bang/collision

    def __init__(self, mode: str = "mock", device_index: int = 0):
        """
        mode: "pyaudio" → real USB microphone
              "mock"    → synthetic events for testing
        """
        self.mode    = mode
        self._queue  = queue.Queue(maxsize=10)
        self._latest = AudioEvent(timestamp=time.time())
        self._lock   = threading.Lock()
        self._running = False

        if mode == "pyaudio":
            self._init_pyaudio(device_index)
        else:
            self._start_mock()

        print(f"✅ Microphone sensor ready (mode={mode})")

    def _init_pyaudio(self, device_index: int):
        try:
            import pyaudio
            self._pa   = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format            = pyaudio.paFloat32,
                channels          = 1,
                rate              = self.SAMPLE_RATE,
                input             = True,
                frames_per_buffer = self.CHUNK_SIZE,
                input_device_index= device_index,
                stream_callback   = self._audio_callback,
            )
            self._stream.start_stream()
            self._running = True

            t = threading.Thread(target=self._process_loop, daemon=True)
            t.start()
            print(f"✅ PyAudio mic opened (device {device_index})")

        except ImportError:
            print("⚠️  pyaudio not installed — pip install pyaudio")
            self._start_mock()
        except Exception as e:
            print(f"⚠️  Mic init failed: {e} — using mock")
            self._start_mock()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — push raw audio to processing queue."""
        import pyaudio
        arr = np.frombuffer(in_data, dtype=np.float32)
        try:
            self._queue.put_nowait(arr)
        except queue.Full:
            pass
        return (None, pyaudio.paContinue)

    def _process_loop(self):
        """Analyse audio chunks from the queue."""
        while self._running:
            try:
                chunk = self._queue.get(timeout=0.5)
                event = self._classify(chunk)
                with self._lock:
                    self._latest = event
            except queue.Empty:
                pass

    def _classify(self, chunk: np.ndarray) -> AudioEvent:
        """Frequency-domain classification of a raw audio chunk."""
        # dBFS loudness
        rms    = np.sqrt(np.mean(chunk ** 2))
        db     = 20 * np.log10(rms + 1e-9)

        # FFT magnitude spectrum
        mag    = np.abs(np.fft.rfft(chunk))
        freqs  = np.fft.rfftfreq(len(chunk), 1 / self.SAMPLE_RATE)

        def band_energy(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            return float(mag[mask].mean()) if mask.any() else 0.0

        horn_e  = band_energy(*self.HORN_BAND)
        siren_e = band_energy(*self.SIREN_BAND)
        total_e = float(mag.mean()) + 1e-6

        event = AudioEvent(db_level=db, timestamp=time.time())

        if db > self.BANG_DB:
            event.event_type = EVENT_BANG
            event.confidence = min(1.0, (db - self.BANG_DB) / 20)
            event.should_stop = True

        elif siren_e / total_e > 0.5:
            event.event_type  = EVENT_SIREN
            event.confidence  = min(1.0, siren_e / total_e)
            event.should_stop = True

        elif horn_e / total_e > 0.4:
            event.event_type   = EVENT_HORN
            event.confidence   = min(1.0, horn_e / total_e)
            event.should_yield = True   # someone honking from behind

        elif total_e > 200:
            event.event_type  = EVENT_CROWD
            event.confidence  = 0.6
            event.should_slow = True

        else:
            event.event_type = EVENT_QUIET

        return event

    def _start_mock(self):
        self.mode     = "mock"
        self._running = True
        t = threading.Thread(target=self._mock_loop, daemon=True)
        t.start()

    def _mock_loop(self):
        """Randomly fire synthetic audio events."""
        import random
        EVENTS = [EVENT_QUIET]*8 + [EVENT_HORN]*3 + [EVENT_SIREN] + [EVENT_CROWD]*2
        while self._running:
            evt   = random.choice(EVENTS)
            event = AudioEvent(
                event_type  = evt,
                confidence  = round(random.uniform(0.6, 0.95), 2),
                db_level    = round(random.uniform(-40, -10), 1),
                timestamp   = time.time(),
                should_stop  = evt in (EVENT_SIREN, EVENT_BANG),
                should_yield = evt == EVENT_HORN,
                should_slow  = evt == EVENT_CROWD,
            )
            with self._lock:
                self._latest = event
            time.sleep(random.uniform(1.5, 4.0))

    def get(self) -> AudioEvent:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if hasattr(self, "_stream"):
            self._stream.stop_stream()
            self._pa.terminate()

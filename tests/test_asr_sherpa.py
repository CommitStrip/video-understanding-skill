import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import asr_sherpa


class StrictASRTests(unittest.TestCase):
    def test_missing_recognizer_never_falls_back(self):
        with self.assertRaises(asr_sherpa.ASRConfigurationError):
            asr_sherpa.transcribe_streaming(None, np.zeros(160, dtype=np.float32))

    def test_offline_stub_fails_explicitly(self):
        with self.assertRaises(NotImplementedError):
            asr_sherpa.transcribe_offline("unused.wav")

    def test_missing_ffmpeg_fails_before_subprocess(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with mock.patch.object(asr_sherpa.shutil, "which", return_value=None):
                with self.assertRaises(asr_sherpa.ASRConfigurationError):
                    asr_sherpa.extract_audio(video.name)

    def test_wav_is_read_in_bounded_chunks(self):
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(np.zeros(16000, dtype="<i2").tobytes())

            chunks = list(asr_sherpa.iter_wav_chunks(path, chunk_sec=0.25))
            self.assertEqual(4, len(chunks))
            self.assertTrue(all(len(samples) == 4000 for _, _, samples in chunks))
            self.assertEqual(0.0, chunks[0][0])
            self.assertEqual(1.0, chunks[-1][1])
        finally:
            os.unlink(path)

    def test_model_directory_must_be_explicit(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(asr_sherpa.ASRConfigurationError):
                asr_sherpa._resolve_model_files(None)


if __name__ == "__main__":
    unittest.main()

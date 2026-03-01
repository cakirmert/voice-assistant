import logging
import numpy as np
from faster_whisper import WhisperModel
import config

logger = logging.getLogger("stt_engine")

class STTEngine:
    """
    Native Windows STT using Faster-Whisper.
    Pinned to SECONDARY_GPU (1660 Super) to save VRAM for LLM.
    """
    
    def __init__(self):
        # Extract device index from "cuda:1" string
        device_index = int(config.SECONDARY_GPU.split(":")[-1])
        logger.info(f"[STT] Initializing Faster-Whisper on GPU {device_index}...")
        
        # Load model explicitly on the secondary GPU
        self.model = WhisperModel(
            config.STT_MODEL_SIZE,
            device="cuda",
            device_index=device_index,
            compute_type="float16" # 1660S works well with FP16
        )
        logger.info(f"[STT] Faster-Whisper ({config.STT_MODEL_SIZE}) loaded successfully.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe audio array to text.
        audio_data: float32 array at 16kHz
        """
        try:
            # audio_data is already 16kHz float32 from AudioCapture
            segments, info = self.model.transcribe(audio_data, beam_size=5)
            
            text = ""
            for segment in segments:
                text += segment.text
                
            return text.strip()
        except Exception as e:
            logger.error(f"[STT] Transcription error: {e}")
            return ""

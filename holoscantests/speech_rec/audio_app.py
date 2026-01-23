"""Holoscan application for audio capture and transcription."""

from holoscan.core import Application, Operator, OperatorSpec
from holoscantests.speech_rec.holoscan_whisper import AudioCaptureOp, WhisperTranscribeOp


class SinkOp(Operator):
    """Dummy sink to consume transcriber output."""
    def setup(self, spec: OperatorSpec):
        spec.input("in")
    
    def compute(self, op_input, op_output, context):
        # Must actually receive the input
        op_input.receive("in")


class AudioTranscriptionApp(Application):
    """Audio capture + Whisper transcription pipeline."""

    def compose(self):
        # Audio capture operator
        audio_capture = AudioCaptureOp(
            self,
            name="audio_capture",
            chunk_duration=0.5,
        )

        # Whisper transcription operator
        transcriber = WhisperTranscribeOp(
            self,
            name="transcriber",
            model_name="large",  # Changed from "small"
            device="cpu",
        )

        # Sink to prevent deadlock
        sink = SinkOp(self, name="sink")

        # Connect pipeline
        self.add_flow(audio_capture, transcriber, {("audio_out", "audio_in")})
        self.add_flow(transcriber, sink, {("done", "in")})


def main():
    app = AudioTranscriptionApp()
    app.run()


if __name__ == "__main__":
    main()
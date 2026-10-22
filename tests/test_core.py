
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import SETTINGS
from dataset import discover_recordings, uncertain_mapping
from evaluator import deterministic_assessment, label_for_score
from media_processing import inspect_media, validate_upload
from schemas import TranscriptSegment, validate_llm_assessment
from transcription import parse_vtt, timestamp_to_seconds, transcribe_deepgram


class CoreTests(unittest.TestCase):
    def valid_assessment(self):
        return {
            "class_label": "Good",
            "score": 70,
            "dimension_scores": {
                key: {
                    "score": 70,
                    "explanation": "Grounded explanation.",
                }
                for key in SETTINGS.rubric_weights
            },
            "strengths": [],
            "improvements": [],
            "evidence": [
                {
                    "observation": "Teacher asks a check question.",
                    "modality": "transcript",
                    "timestamp": 1.0,
                    "excerpt": "Hello class",
                    "frame_reference": None,
                }
            ],
            "counterevidence": [],
            "technical_flags": [],
            "limitations": [],
            "confidence": {
                "level": "low",
                "rationale": "Limited evidence.",
            },
        }

    def test_vtt_timestamps_and_speakers(self):
        segments = parse_vtt(
            "WEBVTT\n\n"
            "1\n"
            "00:00:01.500 --> 00:00:03.000\n"
            "Teacher: Hello class\n"
        )

        self.assertEqual(1.5, segments[0].start)
        self.assertEqual("Teacher", segments[0].speaker)
        self.assertEqual(
            3723.25,
            timestamp_to_seconds("01:02:03.250"),
        )

    def test_malformed_vtt_rejected(self):
        with self.assertRaises(ValueError):
            parse_vtt("WEBVTT\nno cues")

    def test_thresholds_and_weighting(self):
        # Verify the new grade boundaries.
        self.assertEqual("Excellent", label_for_score(85))
        self.assertEqual("Good", label_for_score(70))
        self.assertEqual("Average", label_for_score(60))
        self.assertEqual("Weak", label_for_score(40))
        self.assertEqual("Poor", label_for_score(39.9))

        # Verify scores immediately below the boundaries.
        self.assertEqual("Good", label_for_score(84.9))
        self.assertEqual("Average", label_for_score(69.9))
        self.assertEqual("Weak", label_for_score(59.9))
        self.assertEqual("Poor", label_for_score(0))

        # Verify the weighted calculation.
        scores = {
            "structure_clarity": 80,
            "explanations_examples": 70,
            "checks_feedback": 60,
            "inclusive_interaction": 50,
            "technical_delivery": 90,
        }

        result = deterministic_assessment(scores)

        expected = sum(
            scores[name] * weight / 100
            for name, weight in SETTINGS.rubric_weights.items()
        )

        self.assertEqual(
            round(expected, 1),
            result["final_score"],
        )
        self.assertEqual(
            label_for_score(result["final_score"]),
            result["class_label"],
        )

        # Identical dimension scores should preserve the same overall score.
        uniform_scores = {
            key: 70 for key in SETTINGS.rubric_weights
        }
        uniform_result = deterministic_assessment(uniform_scores)

        self.assertEqual(70.0, uniform_result["final_score"])
        self.assertEqual("Good", uniform_result["class_label"])

    def test_invalid_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.txt"
            path.write_text("x")

            with self.assertRaises(ValueError):
                validate_upload(path, 1)

    def test_empty_upload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.mp4"
            path.touch()

            with self.assertRaises(ValueError):
                validate_upload(path, 1)

    def test_dataset_mapping_flags_weak_as_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weak"
            path.mkdir()
            (path / "B1.mp4").write_bytes(b"x")

            result = discover_recordings(Path(directory))[0]

            self.assertEqual("Poor", result.ordinal_label)
            self.assertTrue(uncertain_mapping(result))
            self.assertFalse(result.mapping_certain)

    def test_invalid_model_output_rejected(self):
        with self.assertRaises(ValueError):
            validate_llm_assessment(
                {"class_label": "Perfect", "score": 101},
                10,
                [],
                set(),
            )

    def test_ungrounded_or_contradictory_model_output_rejected(self):
        transcript = [
            TranscriptSegment(0, 2, "Teacher", "Hello class")
        ]

        assessment = self.valid_assessment()
        assessment["score"] = 10

        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_llm_assessment(
                assessment,
                10,
                transcript,
                set(),
            )

        assessment = self.valid_assessment()
        assessment["evidence"][0]["excerpt"] = "invented quote"

        with self.assertRaisesRegex(ValueError, "not present"):
            validate_llm_assessment(
                assessment,
                10,
                transcript,
                set(),
            )

    def test_external_gate(self):
        if not SETTINGS.allow_external_processing:
            with self.assertRaises(PermissionError):
                transcribe_deepgram(Path("missing.wav"))

    def test_missing_audio_and_video_streams_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x.mp4"
            path.write_bytes(b"x")

            response = SimpleNamespace(
                stdout=json.dumps(
                    {
                        "format": {"duration": "1.0"},
                        "streams": [],
                    }
                )
            )

            with (
                patch(
                    "media_processing.require_binary",
                    return_value="ffprobe",
                ),
                patch(
                    "media_processing.subprocess.run",
                    return_value=response,
                ),
            ):
                result = inspect_media(path)

            self.assertFalse(result["has_audio"])
            self.assertFalse(result["has_video"])

    def test_deepgram_failure_becomes_safe_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.write_bytes(b"wav")

            enabled = SimpleNamespace(
                allow_external_processing=True
            )

            with (
                patch("transcription.SETTINGS", enabled),
                patch.dict(
                    os.environ,
                    {"DEEPGRAM_API_KEY": "test"},
                ),
                patch(
                    "transcription.urllib.request.urlopen",
                    side_effect=TimeoutError,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Deepgram transcription failed",
                ):
                    transcribe_deepgram(path)

    def test_temp_workspace_cleanup(self):
        with tempfile.TemporaryDirectory(
            prefix="classlens_test_"
        ) as directory:
            folder = Path(directory)
            (folder / "private.wav").write_bytes(b"x")

        self.assertFalse(folder.exists())

    def test_missing_opencv_is_actionable(self):
        from video_analysis import sample_frames

        with patch.dict(sys.modules, {"cv2": None}):
            with self.assertRaises(RuntimeError):
                sample_frames(
                    Path("not-a-video.mp4"),
                    Path("frames"),
                )


if __name__ == "__main__":
    unittest.main()

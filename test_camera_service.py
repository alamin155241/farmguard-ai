import time
import unittest
from unittest.mock import patch
import numpy as np
from camera_service import CameraResult, CameraVerifier
from live_test_service import append_live_result, finalize_live_test, persist_completed_live_test

class FakeCapture:
    def __init__(self): self.opened=True
    def isOpened(self): return self.opened
    def read(self): return True,np.zeros((240,320,3),dtype=np.uint8)
    def set(self,*args): return True
    def release(self): self.opened=False

class MultiFrameVerifier(CameraVerifier):
    def __init__(self):
        super().__init__(duration=.12,target_fps=40,prefer_yolo=False); self.calls=0
    def _open_camera(self): return FakeCapture()
    def _detect(self,frame):
        self.calls+=1
        return frame, ([(1,1,20,40,.8)] if self.calls==3 else [])

class CameraTests(unittest.TestCase):
    def test_any_positive_frame_makes_window_positive(self):
        verifier=MultiFrameVerifier(); result=verifier.verify(.12)
        self.assertGreaterEqual(result.frames_processed,3); self.assertTrue(result.person_detected)

    def test_hog_fallback_initializes(self):
        verifier=CameraVerifier(prefer_yolo=False)
        self.assertEqual(verifier.snapshot()["detector"],"OpenCV HOG fallback")

    def test_background_start_is_non_blocking(self):
        verifier=MultiFrameVerifier(); started=time.monotonic(); self.assertTrue(verifier.start(.12))
        self.assertLess(time.monotonic()-started,.05); self.assertTrue(verifier.running)
        verifier._worker.join(1); self.assertFalse(verifier.running)

    def test_live_results_are_separate_from_model_predictions(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"live_test_results.csv"
            append_live_result({"ground_truth":"INTRUSION","rule_state":"CONFIRMED","ml_prediction":"INTRUSION","ml_probability":.9,"camera_result":"PERSON DETECTED","final_result":"VERIFIED INTRUSION","correct":True},path)
            text=path.read_text(encoding="utf-8")
            self.assertIn("VERIFIED INTRUSION",text)
            self.assertFalse((Path(folder)/"predictions.csv").exists())

    def test_live_result_waits_then_persists_across_reruns_once(self):
        import tempfile
        from pathlib import Path
        sensor={"row":{"trial_id":"LIVE-1"},"truth":"INTRUSION","rule":"CONFIRMED","prediction":"INTRUSION","probability":.93,"warnings":[]}
        running={"running":True,"result":None}
        self.assertIsNone(finalize_live_test(sensor,running))
        camera_result=CameraResult(True,None,"12:00:00",detector="test",confidence=.9,frames_processed=20)
        finished={"running":False,"result":camera_result,"frame":None,"detector":"test","frames_processed":20}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"live_test_results.csv"; session={}
            first=persist_completed_live_test(session,sensor,finished,path)
            second=persist_completed_live_test(session,sensor,finished,path)
            self.assertIs(first,session["last_live_test_result"])
            self.assertEqual(second["final_result"],"VERIFIED INTRUSION")
            self.assertEqual(second["correctness"],"CORRECT")
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()),2)

    def test_disagreement_is_preserved_honestly(self):
        sensor={"trial_id":"LIVE-2","truth":"INTRUSION","rule":"WATCH","prediction":"INTRUSION","probability":.7}
        camera=CameraResult(True,None,"12:00:00",detector="test")
        result=finalize_live_test(sensor,{"running":False,"result":camera})
        self.assertEqual(result["rule_state"],"WATCH")
        self.assertEqual(result["camera_result"],"PERSON DETECTED")
        self.assertEqual(result["final_result"],"NOT VERIFIED")

if __name__=="__main__": unittest.main()

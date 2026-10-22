import csv, shutil, subprocess, tempfile, unittest
from pathlib import Path
import pandas as pd
from audit_dataset import audit_dataset
from dataset_service import DatasetStore, ML_FIELDS, aggregate_trial
from ml_inference import predict_trial
from train_model import train

def samples(pir=0,beam=0,ldr=100):
    return [{"pir":pir,"beam_blocked":beam,"ldr":ldr,"gas":200+i,"temperature":30.0,"state":"CONFIRMED" if pir and beam else "NORMAL"} for i in range(16)]

class MLWorkflowTests(unittest.TestCase):
    def test_aggregation_preserves_selected_label(self):
        normal=aggregate_trial(samples(1,1,10),0); intrusion=aggregate_trial(samples(0,0,100),1)
        self.assertEqual(normal["label"],0); self.assertEqual(intrusion["label"],1)
        self.assertEqual(normal["pir_detected"],1); self.assertEqual(intrusion["pir_detected"],0)

    def test_append_audit_refusal_and_temporary_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/"data/ml_dataset.csv"; store=DatasetStore(path)
            with self.assertRaises(ValueError): train(path,root)
            rows=[]
            for i in range(20):
                label=i%2; row=aggregate_trial(samples(label,label,20 if label else 100),label); row["trial_id"]=f"T{i}"; store.append(row); rows.append(row)
            report=audit_dataset(path,root/"data/ml_dataset_clean.csv"); self.assertEqual(report["rows"],20); self.assertFalse(report["cleaned_file_written"])
            metrics=train(path,root); self.assertEqual(metrics["training_samples"],20)
            pred,_=predict_trial(rows[-1],root); self.assertIn(pred,(0,1)); self.assertTrue((root/"esp32_ml_model.h").exists())
            if shutil.which("g++"):
                args=", ".join("0.0f" for _ in metrics["features"])
                source=root/"compile_test.cpp"; source.write_text(f'#include "esp32_ml_model.h"\nint main(){{return predictIntrusion({args});}}\n',encoding="utf-8")
                subprocess.run(["g++","-fsyntax-only",str(source)],cwd=root,check=True,capture_output=True)

if __name__=="__main__": unittest.main()

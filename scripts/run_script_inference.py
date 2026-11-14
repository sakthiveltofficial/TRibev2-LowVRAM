import os
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tribe_local.evaluation.inference import TribeLocalPredictor

os.environ["DEEPSEEK_API_KEY"] = "sk-deda7bed28a74872b353f46fb562f2ae"

with open("tribe_local/configs/local_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

voiceover_text = """Is AI the next dot com crash? Everyone is screaming AI is the future, but look at the numbers.
We are spending billions on GPUs, but the revenue isn't matching the hype.
Companies are buying chips, but are they actually making money back?
History doesn't repeat, but it rhymes. When the hype cools down, the infrastructure costs remain.
Nvidia is printing money, but the guys buying the shovels might be digging their own graves.
Is AI a revolution or a rug pull?"""

print("Running real-time brain-encoding inference on viral script...")
predictor = TribeLocalPredictor(config=cfg, device="cpu")

result = predictor.predict(
    text_input=voiceover_text,
    subject_id=0
)

print("\n" + "=" * 58)
print("  REAL-TIME BRAIN PREDICTION RESULT (fsaverage5)")
print("=" * 58)
print("Stimulus: Viral AI Bubble YouTube Short Script")
print(f"Predicted Cortical Vertices : {result['predicted_bold'].shape[0]} (both hemispheres)")
print(f"Timecourse (TRs)            : {result['predicted_bold'].shape[1]} TRs")
print(f"Inference Latency           : {result['latency_ms']:.2f} ms")
print(f"Max BOLD Intensity          : {result['predicted_bold'].max():.4f}")
print(f"Min BOLD Intensity          : {result['predicted_bold'].min():.4f}")
print(f"Mean BOLD Response          : {result['predicted_bold'].mean():.4f}")
print("=" * 58)

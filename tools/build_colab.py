"""Build a self-contained Colab notebook with no executed/fabricated cloud outputs."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(dict(cell_type='markdown',metadata={},source=source.splitlines(keepends=True)))


def code(source):
    cells.append(dict(cell_type='code',metadata={},source=source.splitlines(keepends=True),execution_count=None,outputs=[]))


md('''# SpikeDistill-RUL: Colab readiness and recovery trial

This is the next milestone, not the full research experiment. Run cells in order.
We prepare FD001, inspect training data, test dependencies, and train a small MLP
on at most 2,048 training windows for two epochs. Official test labels are not used
for model selection. Capped RUL (125 cycles) is the declared pilot target.

**First select Runtime > Change runtime type > an available GPU.** CPU fallback
is supported for debugging; it does not establish GPU compatibility or speed.
Authorize your own Drive mount when prompted. Do not run two sessions against
the same checkpoint directory.

The notebook loads its Python modules from the GitHub repository. Runtime files are temporary;
checkpoints, logs, and plots are written to the project folder in Drive.
''')
code('''import sys, importlib.util, subprocess
# Keep Colab's existing PyTorch/CUDA installation. Install only missing helpers.
if importlib.util.find_spec("torch") is None:
    raise RuntimeError("PyTorch missing. Select a supported Colab runtime before continuing.")
missing = [name for name in ["numpy", "pandas", "matplotlib", "snntorch"]
           if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import snntorch as snn
from snntorch import surrogate
print("Python:", sys.version)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CPU debug mode only; select a GPU before judging training speed.")
print("NumPy:", np.__version__, "pandas:", pd.__version__)
''')
md('''## 1. Connect Drive and choose persistent storage
The uploaded parent folder is named **ML projec** (including the space).
If it is renamed or moved, adjust PROJECT_DRIVE below. The existing dataset
and preparation notebook are left unchanged. This Colab notebook remains in
Colab Notebooks; experiment outputs go inside the project folder.
''')
code('''from google.colab import drive
from pathlib import Path
import json, uuid, shutil, hashlib, os, datetime, platform
drive.mount("/content/drive")
PROJECT_DRIVE = Path("/content/drive/MyDrive/ML projec")
RAW_DRIVE = PROJECT_DRIVE / "CMaps"
OUTPUT_DRIVE = PROJECT_DRIVE / "SpikeDistill-RUL" / "colab_runs"
if not RAW_DRIVE.is_dir():
    raise FileNotFoundError(f"Expected uploaded dataset at {RAW_DRIVE}. Adjust PROJECT_DRIVE.")
OUTPUT_DRIVE.mkdir(parents=True, exist_ok=True)
WORK = Path("/content/spikedistill_work")
WORK.mkdir(parents=True, exist_ok=True)
identity = WORK / "runtime_id.txt"
if not identity.exists():
    identity.write_text(str(uuid.uuid4()))
RUNTIME_ID = identity.read_text().strip()
SESSION_DIR = OUTPUT_DRIVE / "sessions" / RUNTIME_ID
SESSION_DIR.mkdir(parents=True, exist_ok=True)
probe_path = SESSION_DIR / "storage_probe.txt"
probe_path.write_text(RUNTIME_ID)
assert probe_path.read_text() == RUNTIME_ID, "Drive write/read check failed"
environment = dict(runtime_id=RUNTIME_ID, python=sys.version, platform=platform.platform(),
                   torch=str(torch.__version__), cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   numpy=np.__version__, pandas=pd.__version__,
                   snntorch=getattr(snn,"__version__","unknown"))
(SESSION_DIR/"environment.json").write_text(json.dumps(environment,indent=2))
freeze = subprocess.check_output([sys.executable,"-m","pip","freeze"],text=True)
(SESSION_DIR/"requirements-observed.txt").write_text(freeze)
print("Runtime ID:", RUNTIME_ID)
print("Persistent outputs:", OUTPUT_DRIVE)
''')
md('''## 2. Load the project code from GitHub
The first run records its Git commit in Drive. A replacement runtime checks
out that same commit so recovery uses the same code.
''')
code('''REPO_URL = "https://github.com/abhav-arora/ML3103-project.git"
REPO = WORK / "repo"
RUN_DIR = OUTPUT_DRIVE / "fd001_pilot_v1"
RUN_DIR.mkdir(parents=True, exist_ok=True)
revision_file = RUN_DIR / "git_revision.txt"
if not (REPO / ".git").is_dir():
    subprocess.run(["git", "clone", REPO_URL, str(REPO)], check=True)
if revision_file.exists():
    revision = revision_file.read_text().strip()
    if subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=REPO).returncode:
        subprocess.run(["git", "fetch", "origin"], cwd=REPO, check=True)
    subprocess.run(["git", "checkout", "--detach", revision], cwd=REPO, check=True)
else:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    revision_file.write_text(revision + "\\n")
(SESSION_DIR / "git_revision.txt").write_text(revision + "\\n")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
legacy = (REPO / "train_pilot.py").is_file()
pilot_entry = ["train_pilot.py"] if legacy else ["-m", "spikedistill.training.pilot"]
print("Project commit:", revision)
''')
md('''## 3. Verify dependencies and recovery logic
The first command checks leakage-sensitive preprocessing and metric definitions.
The second checks uninterrupted versus checkpoint-resumed training on tiny,
synthetic CPU data, including recovery from a corrupt last checkpoint.
The spiking smoke test checks surrogate backpropagation, learnable leak and threshold
on the selected device. It is not a trained recurrent SNN or the proposed student.
''')
code('''test_names = ["test_prepare_data", "test_metrics", "test_training"]
if not legacy:
    test_names = [f"tests.{name}" for name in test_names]
subprocess.run([sys.executable,"-m","unittest","-v",*test_names],
               cwd=REPO,check=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
projection = torch.nn.Linear(4,8).to(device)
lif = snn.Leaky(beta=0.9,learn_beta=True,learn_threshold=True,
                spike_grad=surrogate.fast_sigmoid()).to(device)
membrane = torch.zeros(2,8,device=device)
loss = 0
for step in range(4):
    spikes,membrane = lif(projection(torch.randn(2,4,device=device)),membrane)
    loss = loss + spikes.mean() + 0.01*membrane.square().mean()
loss.backward()
assert projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
assert lif.beta.grad is not None and torch.isfinite(lif.beta.grad).all()
assert lif.threshold.grad is not None and torch.isfinite(lif.threshold.grad).all()
print("Spiking forward/backward compatibility passed on",device)
''')
md('''## 4. Prepare FD001 using the same fixed protocol
Copy only the three FD001 text files to temporary storage for efficient reads.
Copy their source hashes into the session log. The preprocessing uses 80 training
engines and 20 held-out validation engines, not a random window split. Save the
pipeline metadata in Drive; temporary arrays can be rebuilt after disconnection.
No official test metrics are calculated in this trial.
''')
code('''RAW_LOCAL = WORK/"raw"
RAW_LOCAL.mkdir(exist_ok=True)
source_hashes = {}
for prefix in ["train","test","RUL"]:
    name = f"{prefix}_FD001.txt"
    source = RAW_DRIVE/name
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source,RAW_LOCAL/name)
    source_hashes[name] = hashlib.sha256((RAW_LOCAL/name).read_bytes()).hexdigest()
original_manifest = PROJECT_DRIVE/"SpikeDistill-RUL"/"artifacts"/"data"/"FD001"/"split_manifest.json"
if original_manifest.is_file():
    original = json.loads(original_manifest.read_text())
    for prefix in ["train","test","RUL"]:
        assert source_hashes[f"{prefix}_FD001.txt"] == original["sources"][prefix]["sha256"], "Dataset changed since local preparation"
if legacy:
    from prepare_data import prepare_subset, load_trajectories
else:
    from spikedistill.data.prepare import prepare_subset, load_trajectories
DATA_LOCAL = WORK/"prepared"
audit = prepare_subset(RAW_LOCAL,DATA_LOCAL,"FD001",window=30,cap=125,seed=42)
DATA_FD001 = DATA_LOCAL/"FD001"
for name in ["preprocessing.json","split_manifest.json","audit.json"]:
    shutil.copy2(DATA_FD001/name,SESSION_DIR/name)
(SESSION_DIR/"source_hashes.json").write_text(json.dumps(source_hashes,indent=2))
print(json.dumps(audit,indent=2))
''')
md('''## 5. Exploratory analysis: what do training engines look like?
Plots use only the training-engine partition. We inspect lifetime variation, label
distribution, several sensor histories, and sensor correlations. Observed patterns
are diagnostic; they do not establish model performance or causality.
''')
code('''manifest = json.loads((DATA_FD001/"split_manifest.json").read_text())
prep = json.loads((DATA_FD001/"preprocessing.json").read_text())
all_train = load_trajectories(RAW_LOCAL/"train_FD001.txt")
training = all_train[all_train.engine_id.isin(manifest["train_engine_ids"])].copy()
with np.load(DATA_FD001/"train.npz",allow_pickle=False) as a:
    train_y = a["y"].copy()
fig, axes = plt.subplots(1,2,figsize=(11,4))
training.groupby("engine_id").cycle.max().hist(ax=axes[0],bins=15)
axes[0].set(title="Training engine lifetimes",xlabel="Cycles to failure",ylabel="Engines")
axes[1].hist(train_y,bins=20)
axes[1].set(title="Training window labels (cap 125)",xlabel="Remaining life (cycles)",ylabel="Windows")
fig.tight_layout(); fig.savefig(SESSION_DIR/"eda_lifetimes_labels.png",dpi=160); plt.show()
chosen = [s for s in ["sensor_02","sensor_07","sensor_11"] if s in prep["sensors"]]
fig,axes = plt.subplots(len(chosen),1,figsize=(10,3*len(chosen)),squeeze=False)
for sensor,ax in zip(chosen,axes[:,0]):
    for engine in manifest["train_engine_ids"][:3]:
        rows = training[training.engine_id==engine]
        ax.plot(rows.cycle,rows[sensor],label=f"Engine {engine}")
    ax.set(title=sensor,xlabel="Observed cycle",ylabel="Raw sensor value"); ax.legend()
fig.tight_layout(); fig.savefig(SESSION_DIR/"eda_sensor_histories.png",dpi=160); plt.show()
correlation = training[prep["sensors"]].corr()
fig,ax = plt.subplots(figsize=(9,8))
image = ax.imshow(correlation,vmin=-1,vmax=1,cmap="coolwarm")
ax.set_xticks(range(len(correlation)),correlation.columns,rotation=90)
ax.set_yticks(range(len(correlation)),correlation.index)
ax.set_title("Sensor correlations: training engines only")
fig.colorbar(image,ax=ax); fig.tight_layout()
fig.savefig(SESSION_DIR/"eda_correlations.png",dpi=160); plt.show()
''')
md('''## 6. Sanity baseline: always predict the training-label mean
This is not a neural model. It gives us a simple reference on held-out validation
engines. All metrics below use capped targets and are in cycles where applicable.
The asymmetric score penalizes optimistic remaining-life predictions more heavily.
''')
code('''from metrics import regression_metrics
with np.load(DATA_FD001/"validation.npz",allow_pickle=False) as a:
    validation_y = a["y"].copy()
mean_prediction = np.full_like(validation_y,train_y.mean())
reference = regression_metrics(validation_y,mean_prediction)
(SESSION_DIR/"mean_baseline_validation.json").write_text(json.dumps(reference,indent=2))
print("Validation metrics, capped RUL; NOT official test results:")
print(json.dumps(reference,indent=2))
''')
md('''## 7. First training pass: two-epoch compatibility pilot
This uses a small feed-forward network with 2,048 randomly selected training
windows. It is deliberately short. It is **not** a final baseline result.
Checkpoint files are immutable per epoch; previous files are kept. Each save is
read back before updating the latest-file index. This reduces interruption risk
without promising that remote storage cannot fail.

Re-running this cell resumes an existing compatible run; it does not reset it.
Do not change its settings while reusing the same run directory.
''')
code('''RUN_DIR = OUTPUT_DRIVE/"fd001_pilot_v1"
command = [sys.executable,*pilot_entry,
           "--data-dir",str(DATA_FD001),"--run-dir",str(RUN_DIR),
           "--epochs","2","--max-examples","2048","--runtime-id",RUNTIME_ID]
subprocess.run(command,cwd=REPO,check=True)
progress = json.loads((RUN_DIR/"progress.json").read_text())
print(json.dumps(progress,indent=2))
''')
md('''## 8. Stop here and test a real runtime replacement

1. Confirm the prior cell reports a verified epoch-2 checkpoint in Drive.
2. Use Colab's runtime controls to disconnect and delete the runtime (wording can vary).
   Simply reconnecting the browser to the same runtime does not test persistence.
3. Start a fresh GPU runtime and rerun the preceding cells. Re-authorize Drive if prompted.
   The pilot cell should load its checkpoint, not repeat completed epochs.
4. Run the cell below. It refuses to mark a real-restart check as passed if the
   temporary runtime identity has not changed. It restores model and optimizer
   state, checks saved predictions, then trains the next epoch.

Cross-device/version numerical equality is not guaranteed. A restored-prediction
check failure is a reason to investigate rather than silently restart training.
''')
code('''if legacy:
    from train_pilot import load_latest
else:
    from spikedistill.training.pilot import load_latest
checkpoint,checkpoint_path = load_latest(RUN_DIR)
if checkpoint is None or checkpoint["epoch"] < 2:
    raise RuntimeError("Finish the two-epoch pilot first.")
if checkpoint["origin_runtime_id"] == RUNTIME_ID:
    print("Real restart NOT verified yet. Follow the runtime replacement instructions above.")
else:
    resume_command = command.copy()
    resume_command[resume_command.index("--epochs")+1] = "3"
    subprocess.run(resume_command,cwd=REPO,check=True)
    latest = json.loads((RUN_DIR/"progress.json").read_text())
    assert latest["completed_epoch"] >= 3
    evidence = dict(real_runtime_replacement_verified=True,origin_runtime_id=latest["origin_runtime_id"],
                    restored_in_runtime=RUNTIME_ID,completed_epoch=latest["completed_epoch"],
                    device=latest["device"],torch_version=latest["torch_version"])
    (RUN_DIR/"recovery_verified.json").write_text(json.dumps(evidence,indent=2))
    print("Checkpoint recovery verified in a different runtime:")
    print(json.dumps(evidence,indent=2))
''')
md('''## 9. Review pilot progress and next decision
These curves describe only the compatibility pilot. Three epochs cannot establish
research quality or the full experiment budget. Epoch timing excludes checkpoint
upload overhead, which still affects wall-clock cost.
After the recovery test passes on a GPU, review timing before scheduling full
baselines. Spiking/recurrent models can be much slower than this small MLP.
''')
code('''progress = json.loads((RUN_DIR/"progress.json").read_text())
history = progress["history"]
if history:
    table = pd.DataFrame([dict(epoch=r["epoch"],train_normalized_mse=r["train_normalized_mse"],
                              validation_RMSE=r["validation"]["RMSE"],seconds=r["seconds"]) for r in history])
    display(table)
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot(table.epoch,table.validation_RMSE,marker="o",label="MLP pilot validation")
    ax.axhline(reference["RMSE"],color="grey",linestyle="--",label="Mean baseline")
    ax.set(xlabel="Completed epoch",ylabel="Capped validation RMSE (cycles)",title="Compatibility pilot only")
    ax.legend(); fig.tight_layout(); fig.savefig(SESSION_DIR/"pilot_validation.png",dpi=160); plt.show()
evidence = RUN_DIR/"recovery_verified.json"
print("Real-runtime recovery evidence:",evidence.exists())
print("Persistent run folder:",RUN_DIR)
print("Save this notebook (Ctrl+S) so cell outputs are available for review.")
''')

# Compile ordinary Python code cells before publishing; no cloud outputs invented.
for index,cell in enumerate(cells):
    if cell['cell_type']=='code':
        ast.parse(''.join(cell['source']),filename=f'colab_cell_{index}')
notebook = dict(nbformat=4,nbformat_minor=0,cells=cells,
                metadata=dict(colab=dict(name='ML_project.ipynb',provenance=[]),
                              kernelspec=dict(name='python3',display_name='Python 3'),
                              language_info=dict(name='python'),accelerator='GPU'))
(ROOT/'notebooks'/'ML_project.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
print(f'Built ML_project.ipynb: {len(cells)} cells; all Python cells compile; outputs intentionally empty')

"""Build the CPU GUI as one Windows executable."""
from pathlib import Path
import os
import sys

from PyInstaller.__main__ import run as pyinstaller_run

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'algorithm'

def main():
    models = APP / 'models'
    required = ('alignment.npz', 'alignment_calibration.joblib', 'existing_calibration.joblib', 'vision.pt')
    missing = [name for name in required if not (models / name).is_file()]
    if missing:
        raise SystemExit('Missing private model files in algorithm/models: ' + ', '.join(missing))
    work = ROOT / 'build' / 'cpu-single'
    output = ROOT / 'dist-cpu-single'
    work.mkdir(parents=True, exist_ok=True)
    profile = work / 'distribution.json'
    profile.write_text('{"device":"cpu"}\n', encoding='utf-8')
    tcl = Path(sys.base_prefix) / 'tcl'
    for variable, folder in (('TCL_LIBRARY', 'tcl8.6'), ('TK_LIBRARY', 'tk8.6')):
        path = tcl / folder
        if path.is_dir():
            os.environ[variable] = '//?/' + str(path.resolve()).replace('\\', '/')
    pyinstaller_run([
        '--noconfirm', '--clean', '--onefile', '--windowed',
        '--name', 'Autoto-CPU',
        '--distpath', str(output), '--workpath', str(work / 'pyinstaller'),
        '--add-data', f'{models};models', '--add-data', f'{profile};.',
        '--collect-all', 'tkinterdnd2', '--collect-all', 'pyzmq',
        str(APP / 'v1_oto.pyw'),
    ])

if __name__ == '__main__':
    main()

import os
from pathlib import Path
import platform
import sys
import tempfile


if getattr(sys, 'frozen', False) and 'CUPY_CACHE_DIR' not in os.environ:
    cache = Path(tempfile.gettempdir()) / 'UTAU-Oto-Automation' / 'cupy-cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = str(cache)

if getattr(sys, 'frozen', False):
    cuda_bin = Path(sys._MEIPASS) / 'bin'
    if cuda_bin.is_dir():
        os.environ['CUDA_PATH'] = str(Path(sys._MEIPASS))
        os.environ['PATH'] = str(cuda_bin) + os.pathsep + os.environ.get('PATH', '')
        _cuda_dll_directory = os.add_dll_directory(str(cuda_bin))


def _prepare_windows_tcl() -> None:
    """Avoid Tcl's path normalization failure under some Windows profiles."""
    if sys.platform != 'win32' or not getattr(sys, 'frozen', False):
        return
    base = Path(sys._MEIPASS)
    def native(path: Path) -> str:
        return '//?/' + str(path).replace('\\', '/')
    os.environ['TCL_LIBRARY'] = native(base / '_tcl_data')
    os.environ['TK_LIBRARY'] = native(base / '_tk_data')
    machine = os.environ.get('PROCESSOR_ARCHITECTURE', platform.machine()).lower()
    arch = {'amd64': 'win-x64', 'x86': 'win-x86', 'arm64': 'win-arm64'}.get(machine)
    if arch:
        dnd = base / 'tkinterdnd2' / 'tkdnd' / arch
        os.environ['TCLLIBPATH'] = '{' + native(dnd) + '}'


_prepare_windows_tcl()

# PyInstaller stores Python modules in an archive, so Numba cannot locate
# their .py files when librosa asks for a persistent JIT cache.  Keep JIT
# compilation enabled, but use its in-memory cache inside frozen builds.
if getattr(sys, 'frozen', False):
    from numba.core.dispatcher import Dispatcher
    from numba.core.ccallback import CFunc
    from numba.np.ufunc.ufuncbuilder import UFuncDispatcher
    Dispatcher.enable_caching = lambda self: None
    CFunc.enable_caching = lambda self: None
    UFuncDispatcher.enable_caching = lambda self: None

    from utau_oto.v1_ui import main

if getattr(sys, 'frozen', False):
    _app_name = Path(sys.executable).name
    if _app_name == 'Autoto-CUDA.exe':
        os.environ.setdefault('AUTOTO_DEVICE', 'cuda')
    elif _app_name == 'Autoto-CPU.exe':
        os.environ.setdefault('AUTOTO_DEVICE', 'cpu')
    elif _app_name == 'Autoto.app':
        os.environ.setdefault('AUTOTO_DEVICE', 'cpu')

if __name__ == '__main__':
    if len(sys.argv)>=3 and sys.argv[1] in ('--self-test','--self-test-cuda'):
        from utau_oto.experimental.parallel import run
        from utau_oto.writer import serialize
        rows,reports=run(sys.argv[2],device='cuda' if sys.argv[1]=='--self-test-cuda' else 'cpu',
                   methods=('existing','vision','alignment'),
                   return_data=True,cache_enabled=False)
        if not rows or not serialize(rows) or any(report['errors'] for report in reports.values()):
            raise SystemExit(1)
    else:main()

# Third-party notices

LoRA Factory itself is distributed under the MIT License in `LICENSE`. Third-party
components remain under their own terms. Exact Factory versions are resolved in `uv.lock`;
exact managed-runtime versions are recorded in `runtime-lock.txt` and
`backend-manifest.json`.

The Windows build exports the license and notice files present in every bundled Python
distribution to the adjacent `licenses/` directory. `PACKAGE-METADATA.txt` in each
versioned subdirectory records the installed package name, version, and declared license
metadata. This summary does not replace those files or the upstream license terms.

## Windowed Factory distribution

### CPython

- Source and license: <https://docs.python.org/3.12/license.html>
- Version: `3.12.13`
- License: Python Software Foundation License Version 2 and incorporated notices

The build exports the `LICENSE.txt` that accompanies the exact CPython interpreter embedded
by PyInstaller.

### Qt for Python / PySide6 / Shiboken6

- Source and licensing overview: <https://doc.qt.io/qtforpython-6/>
- Qt open-source obligations: <https://www.qt.io/development/open-source-lgpl-obligations>
- Resolved version: PySide6/Shiboken6 `6.11.1` (recorded in `uv.lock`)
- Available licenses: LGPL-3.0-only, GPL-2.0-only, GPL-3.0-only, or applicable Qt commercial terms
- Release route: LGPL-3.0-only; full LGPL/GPL texts are in `licenses/Qt/`

The one-dir build keeps the Qt/PySide shared libraries as separate files. Do not merge,
obfuscate, or prevent replacement of those libraries when distributing under LGPL terms.
The exact release procedure, upstream source links, replacement expectations, and Qt
third-party attribution references are in `licenses/Qt/README.md`. Anyone redistributing
a build is responsible for meeting the selected license's notice, source,
relinking/replacement, and other requirements.

The GUI uses only QtCore, QtGui, QtNetwork, and QtWidgets. Unused Qt PDF, QML,
Quick, and Virtual Keyboard modules/plugins are excluded from the v0.1
PyInstaller output. This avoids redistributing the GPLv3-only Qt Virtual
Keyboard module and the additional Qt PDF/PDFium attribution payload without a
feature that needs them. Any future use of those modules requires a new
license review and updated notices.

### PyInstaller

- Source: <https://pyinstaller.org/>
- License: GPL-2.0-or-later with the PyInstaller bootloader special exception

The special exception permits distribution of applications produced with PyInstaller;
the exact `COPYING.txt` supplied by the installed build tool is exported to `licenses/`.

### Microsoft Visual C++ runtime

- Redistributable terms and current downloads: <https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist>

The Windows build can include Microsoft Visual C++ runtime DLLs required by CPython and
native extension modules. Those DLLs remain subject to the applicable Microsoft Visual
Studio redistributable terms; the LoRA Factory license does not replace those terms.
The exact release-builder check and official links are in
`licenses/Microsoft-Visual-Cpp/README.md`.

The public v0.1 GitHub ZIP uses `SystemVcRuntime` mode and intentionally does not
redistribute those Microsoft DLLs. Users must install the official Microsoft Visual C++
x64 Redistributable separately before launching it. A locally bundled self-use build is
not a public redistribution unless the build operator has verified the applicable
Microsoft redistribution rights.

### Bundled Python packages

The GUI build includes the following direct packages and the transitive packages selected
by the lockfile:

- Alembic, Mako, SQLAlchemy, greenlet, Pydantic, annotated-types, pydantic-core,
  typing-inspection, PyYAML, Rich, markdown-it-py, mdurl, tomlkit, Typer, annotated-doc,
  and colorama: MIT-family licenses as declared by each distribution.
- ImageHash and Pygments: BSD-2-Clause family licenses.
- psutil and MarkupSafe: BSD-3-Clause family licenses.
- Pillow: MIT-CMU.
- safetensors: Apache-2.0.
- shellingham: ISC.
- structlog: MIT or Apache-2.0.
- typing-extensions: PSF-2.0.
- setuptools and its vendored modules: MIT and the component-specific licenses exported
  from the installed distribution.
- packaging: Apache-2.0 or BSD-2-Clause.
- NumPy, SciPy, and PyWavelets: their primary BSD/MIT-family licenses plus notices for
  bundled components such as OpenBLAS, LAPACK, and applicable compiler runtimes.

Use the generated `licenses/` directory as the authoritative notice bundle for the exact
Windows build; scientific wheels can carry additional component-specific terms.

## Separately installed managed backend

The managed backend is not embedded in the GUI executable. It is installed into a
versioned directory under `%LOCALAPPDATA%\LoRAFactory\` from the hash-verified manifest and
lock. A prebuilt managed runtime must not be redistributed without separately collecting
and satisfying the licenses of every package in `runtime-lock.txt`.

### kohya-ss/sd-scripts

- Source: <https://github.com/kohya-ss/sd-scripts>
- Pinned release: `v0.11.1`
- Pinned commit: `6721028c79ee85a78b3a06dfd8954dae310a1cce`
- License: Apache License 2.0

The installer obtains the unmodified pinned source and installs it editable with dependency
resolution disabled. Dependencies come only from the exact-version runtime lock.

### WD EVA02 Large Tagger v3

- Source: <https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3>
- Pinned release/revision: `v1.0` / `c5303bb7139430db980e4c680a778fe79d72b541`
- License: Apache License 2.0

Model files are downloaded separately and verified against the manifest. They are not
bundled into the GUI executable.

### OpenAI CLIP ViT-L/14 image embedding model

- Source: <https://huggingface.co/openai/clip-vit-large-patch14>
- Pinned revision: `32bd64288804d66eefd0ccbe215aa642df71cc41`
- License: MIT

The vision config and projection weights are installed separately, verified by SHA-256,
and executed only in the isolated managed CUDA runtime. The physical `clip_embed.py`
launcher is bundled for that managed process, but the approximately 1.71 GB model file is
not bundled into the GUI executable.

### PyTorch and torchvision

- Source: <https://pytorch.org/>
- Managed profile: PyTorch 2.13.0 and torchvision 0.28.0, official CUDA 13.0 wheels
- License: BSD-3-Clause

### ONNX Runtime

- Source: <https://onnxruntime.ai/>
- Managed profile: onnxruntime-gpu 1.28.0
- License: MIT

### NVIDIA CUDA runtime

CUDA runtime components remain subject to the NVIDIA CUDA Toolkit End User License
Agreement and redistributable component terms. The managed runtime uses officially
published binary wheels; the GUI executable does not embed the system CUDA Toolkit.

The managed lock also contains `pytorch-optimizer`, whose package metadata is Apache-2.0
but whose documentation identifies component-level non-commercial terms for some
algorithms. The default Factory profiles use `AdamW` or `AdamW8bit`; do not publish a
prebuilt managed runtime or advertise unrestricted commercial use without reviewing every
package and component in the lock and exporting their exact notices.

## User-selected base models

SDXL and Illustrious-derived checkpoints have model-specific licenses. LoRA Factory records
the selected file hash but does not infer or grant a license from its architecture. Users
must review the checkpoint's actual license and usage terms, as well as rights to the
training images and any generated outputs. The pinned CLIP model card also documents
research-oriented intended use and deployment limitations; the model's source license
does not remove those usage warnings.

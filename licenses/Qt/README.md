# Qt for Python redistribution material

LoRA Factory v0.1 bundles PySide6/Shiboken6 and uses the Qt open-source LGPLv3
route for this release. No Qt commercial license is granted or implied by this
repository.

The exact PySide6/Shiboken6 version is recorded in `uv.lock`. The distribution
contains the following license texts:

- `LGPL-3.0-only.txt`: the LGPL terms used for the Qt open-source route.
- `GPL-3.0-only.txt`: the GPL terms incorporated by the LGPL text.

The v0.1 GUI imports only QtCore, QtGui, QtNetwork, and QtWidgets. The
PyInstaller specification explicitly excludes unused Qt PDF, QML, Quick, and
Virtual Keyboard modules and their native plugins. This matters because Qt
Virtual Keyboard is GPLv3-only, and Qt PDF carries additional PDFium and
third-party attribution requirements. A later feature that uses any of these
modules requires a fresh license review before distribution.

The Windows build is PyInstaller one-dir. Qt/PySide native libraries remain
separate files under `_internal/PySide6/` and `_internal/shiboken6/`; the
build does not merge or UPX-compress them. Do not replace this layout with a
single opaque archive or otherwise prevent users from replacing compatible
Qt/PySide libraries.

For the exact Qt/PySide source and third-party attribution material, use the
versioned upstream sources and the official Qt licensing pages:

- <https://doc.qt.io/qtforpython-6/>
- <https://www.qt.io/development/open-source-lgpl-obligations>
- <https://www.qt.io/development/download-open-source>
- <https://download.qt.io/official_releases/qt/>

A recipient who modifies the Qt libraries must be able to obtain the
corresponding source and rebuild or replace the modified compatible libraries.
This file records the release procedure; it is not a substitute for the
license terms above or legal advice.

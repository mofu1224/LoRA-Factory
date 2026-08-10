# Microsoft Visual C++ runtime notice

The Windows ZIP may contain Microsoft Visual C++ runtime DLLs required by
CPython and native extension modules, including `VCRUNTIME140*.dll` and
`MSVCP140*.dll`.

These files are not covered by the LoRA Factory MIT license. Their
redistribution is subject to the applicable Microsoft Visual Studio license
terms and Microsoft's redistributable-file list.

- <https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170>
- <https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>
- <https://visualstudio.microsoft.com/license-terms/>

The release builder must verify that the Visual Studio/Build Tools license
under which the native runtime files were obtained permits redistribution.
Do not publish a build containing these DLLs if that entitlement has not been
verified.

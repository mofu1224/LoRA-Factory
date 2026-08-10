# LoRA Factory

LoRA Factory は、Windows 11 向けのGUIツールです。  
手元の画像から **SDXL / Illustrious 互換のLoRA** を作成する流れを一元化します。

- 画像取り込み
- 検査・前処理
- WD14 タグ付け
- 学習実行
- LoRA の比較・採点・輸出

---

## 使い方（5分で開始）

### 1) 利用条件
- Windows 11 (x64)
- NVIDIA GPU（学習時）
- SDXL/Illustrious 互換 `.safetensors` ベースモデル
- Codex CLI が使える ChatGPT アカウント（任意）
- Microsoft Visual C++ x64 Redistributable（公開ZIPの起動前に導入）
- GitHub Release 版を展開する場合は、`LoRA Factory.exe` が入ったZIPを展開

Microsoft公式のx64ランタイムは以下から入手してください。公開ZIPには
MicrosoftのランタイムDLLを同梱していません。

<https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170>

### 2) GUI起動
1. `LoRA Factory.exe` を起動
2. Setup の確認を実行
3. 「Managed Training Runtime」の `Repair` を実行して初期環境を構築
4. `New Project` でプロジェクト作成 → Dataset確認 → `Create LoRA`

※ CUDA/PyTorch/sd-scripts/WD14/CLIP はアプリ本体には同梱しません。  
初回起動時にローカル管理ランタイムとしてインストールします。

### 3) 開発者（ソース実行）

```powershell
.\scripts\bootstrap.ps1
.\scripts\run_dev.ps1
```

必要に応じて `-Console` を付けてコンソール表示付きで起動します。

---

## 設計の前提

- `dataset/raw/` は原則変更しない（破壊しない）
- 画像は検証付きコピーで管理し、派生データとして保存
- 原画像や重み、トークン情報は API へ不用意に送らない
- 監査ログや診断情報は最小化し、外部に秘匿情報が残らないよう制御

---

## 主なチェックコマンド

```powershell
uv run --frozen lora-factory doctor
uv run --frozen lora-factory fake-e2e
uv run --frozen lora-factory inspect-model "D:\Models\base.safetensors"
```

---

## リリースとビルド

```powershell
.\scripts\build_windows.ps1
```

GitHub公開用の配布ビルドは、Microsoft DLLを同梱しない次のモードで作成します。

```powershell
.\scripts\build_windows.ps1 -SystemVcRuntime
```

`dist\windows-<timestamp>\LoRA Factory\` に one-dir 版を出力します。  
`licenses\`、QtのLGPL/GPL全文、Qt配布手順、Microsoft Visual C++ランタイム通知を
同梱します。配布前に `THIRD_PARTY_NOTICES.md` と `licenses\` を確認してください。

---

## 参考情報

- [docs/user-guide-ja.md](./docs/user-guide-ja.md)
- [docs/developer-guide.md](./docs/developer-guide.md)
- [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md)

## ライセンス

本体は [MIT License](./LICENSE) です。

PySide6/Qt、CPython、PyInstaller、各Pythonパッケージ、Microsoft Visual C++ランタイム、
CUDA、モデル、ユーザーが選択するベースモデルはそれぞれの条件に従います。

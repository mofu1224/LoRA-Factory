# Codex CLI実行環境非依存タイムアウト設計

## 目的

Runtime Codexを、初期状態のCodex CLI、CodexRouterを経由するCLI、設定が古い・壊れているCLI、CLIが未インストールの環境で同じ安全な契約で扱う。目的は「外部サービスが必ず成功する」ことではなく、無限待ち・pipe詰まり・再試行による時間超過をなくし、失敗を復旧可能な状態で返すことである。

## 実行プロファイル

CodexRuntimeAdapterが起動時にCLI実行ファイルとユーザー設定を読み取り、次の順にプロファイルを作る。

| プロファイル | 判定 | 動作 |
| --- | --- | --- |
| direct-configured | 通常のCodex設定、または設定なし | 既定のユーザー設定を使い、Factoryが抽出した全MCPを個別に無効化 |
| router-configured | base_url、model catalog、provider名などにRouterの印がある | Router設定を保持したまま実行 |
| direct-isolated | Router/未知プロファイルの失敗後 | --ignore-user-configでユーザー設定を読み込まずに再試行。認証状態はCLIのCODEX_HOME境界に残す |
| unavailable | 実行ファイルなし、必須オプションなし、画像入力非対応 | CLIを起動せず、通常reviewは決定論的fallback、Dataset refinementはrecoverable failure |

設定ファイルは最大256 KiBまで、構造とMCP名だけを読み取る。URL、トークン、パス、provider値は監査ログへ保存しない。helpプローブが成功した場合だけ、実際に提供されるオプションへ絞り込む。プローブ不能時は既知の安全なオプションを使い、起動失敗を次のプロファイルへ渡す。

## タイムアウト状態機械

1. Popen(shell=False)後、stdout/stderrを専用readerで継続排出する。
2. 初回出力までのstartup timeout（既定45秒）を監視する。
3. 出力後は無通信のidle timeout（既定120秒）を監視する。
4. 全試行を含むtotal timeout（既定180秒）を1つだけ消費する。再試行ごとに180秒を再付与しない。
5. turn.completed等の完了eventとoutput-last-messageの有効なJSONを確認したら、短いgrace後にwrapperを終了させて結果を検証する。
6. timeout、cancel、完了回収のいずれもProcessTreeへ終了要求を出し、bounded wait後にkillする。子プロセスを残さない。

監査には総時間超過、起動超過、無通信超過、完了event観測、選択プロファイルを保存する。stderr本文や環境値は保存しない。

## CLI引数と安全境界

各呼び出しはargv配列で組み立て、--ephemeral --sandbox read-only --json --output-schema --output-last-message --cdを必須とする。画像はCLIの--imageへ直接渡し、画像付きでもnode_replを含む既知・設定済みMCPをすべて無効化する。apps、browser、computer、multi-agent、plugins、shell toolも無効化する。

ユーザー設定を無視する代替プロファイルはRouter/未知環境の失敗後だけ使う。FactoryはユーザーのCodex設定を書き換えず、認証情報をプロジェクトへコピーしない。

## fallbackと復旧

Dataset refinementは品質・caption整合性を学習前に保証する境界なのでfallbackしない。全プロファイルが失敗した場合はauditを保存してrecoverable failureで停止し、Codex復旧後に同じrunをResumeする。Dataset以外は既存のallow_without_codex設定に従って決定論的fallbackを選べる。

## 検証マトリクス

- 設定なしのDirect CLI: executable解決、必須引数、read-only実行
- CodexRouter設定: Routerプロファイル、MCP無効化、失敗時のisolated再試行
- 壊れた設定: UNKNOWN判定、設定値を出力しない、isolated再試行
- PATH外のWindowsインストール: LocalAppData/AppData既知パスの解決
- 大量stdout/stderr: pipe readerによるデッドロック回避
- 起動停止・無通信・総時間超過: 各フラグとProcessTree終了
- 完了event先行: wrapper終了待ちを省略し、schema検証へ進む
- cancel: retryを止め、auditを付けて即時終了

CLIの--config、--ignore-user-config、--image、--json、--output-schema、--output-last-message、--sandboxなどの仕様は[Codex CLI developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)を参照する。

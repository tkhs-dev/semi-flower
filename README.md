## 環境構築
環境をWSL2上に構築します.
1. このリポジトリをクローン
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2. uvのインストール
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
3 必要なソフトウェアのインストール
    ```bash
    sudo apt install -y libffi-dev
    sudo apt install -y libsqlite3-dev
    sudo apt install libbz2-dev
    sudo apt install liblzma-dev
    sudo apt install libreadline-dev
    sudo apt install libncurses5-dev
    ```
4. Pythonの設定
    ```bash
    uv python install 3.13.11
    ```
   
5. 仮想環境の作成
    ```bash
    uv venv -p 3.13.11 venv
    ```
6. 仮想環境の有効化
      ```bash
      source venv/bin/activate
      ```
7. GPUを使用する場合のみ
    ```bash
    pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
    ```
8. 必要なパッケージのインストール
    ```bash
    pip install -e .
    ```
以降、ターミナルを開き直すたびに5.仮想環境の有効化のみを実行する必要がある。それ以外は最初の一回のみでよい。
## 使い方
### GPUを使いたい
pyproject.tomlの`[tool.flwr.federations]`の`default`を`local-simulation-gpu`に変更する。

### 集約方法のカスタマイズ
`pytorchexample/strategy.py`の`get_strategy`関数が任意のStrategyを返すように編集することで、集約方法をカスタマイズできる。

### 悪意のあるクライアントの導入
`pyproject.toml`の`[tool.flwr.app.config]`の`malicious-nodes`の数値を変更することで、悪意のあるクライアントの数を変更できる。
悪意のあるクライアントは、学習時にラベルを一つずらして学習を行うようになっている。
この動作は`pytorchexample/client.py`の`is_malicious`のif文内で定義されている。

### 学習の実行
   ```bash
   flwr run .
   ```
学習の完了後,

### 学習済みモデルでの推論の実行
まず、推論したい画像を28x28ピクセルのグレースケール画像として用意し、カレントディレクトリに配置する。
その後、次のコマンドを実行する。
   ```bash
   python infer.py --model <モデルファイル名> --image <画像ファイル名>
   ```
### モデルの解析
データセットに対する誤答などの情報を解析して表示する。
   ```bash
    python analyze.py --model <モデルファイル名> --topk <表示する数>
   ```

### サンプルデータ
`sample.pt`: 学習済みモデルのサンプルデータ  
`sample.jpg`: 推論用のサンプル画像(A)

   ```bash
   # サンプル画像の推論をテスト
   python infer.py --model sample.pt --image sample.jpg
   ```
で推論可能

### デフォルトの設定など
```yaml
num-server-rounds = 5 # サーバーのラウンド数
fraction-evaluate = 1.0 # 評価に参加するクライアントの割合
local-epochs = 5 # 各クライアントのローカルエポック数
learning-rate = 0.01 # 学習率
batch-size = 64 # バッチサイズ
```
この設定で、RTX 5080を搭載したマシンで約20分で学習が完了します。
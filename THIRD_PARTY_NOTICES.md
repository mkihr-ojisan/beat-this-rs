# 第三者の著作権表示

本リポジトリには、下記のソフトウェアを Rust に移植したコードが含まれる。

モデルの重みは本リポジトリに含まれない。重みの利用条件は配布元の定めに従う。

## Beat This!

- URL: https://github.com/CPJKU/beat_this
- 対象: ライセンスを確認した時点の `main`（コミット `b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c`）。移植時のコミットは記録されていない。
- 移植したコード:
  - `beat_this/model/beat_tracker.py` → `src/model/mod.rs`
  - `beat_this/model/roformer.py` → `src/model/roformer.rs`
  - `beat_this/model/postprocessor.py` の `Postprocessor`（`minimal`）と `deduplicate_peaks` → `src/postprocess.rs`
  - `beat_this/inference.py` の `split_piece`、`aggregate_prediction`、`split_predict_aggregate` → `src/chunking.rs`、`src/tracker.rs`
  - `beat_this/preprocessing.py` の `LogMelSpect` のパラメーターと処理手順 → `src/mel.rs`

```text
MIT License

Copyright (c) 2024 Institute of Computational Perception, JKU Linz, Austria

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## torchaudio

- URL: https://github.com/pytorch/audio
- 対象: タグ `v2.3.1`（コミット `3edcf69e78a3c9a3077a11159861422440ec7d4a`）。ライセンスを確認した時点の Beat This! の `requirements.txt` で指定されているバージョンである。
- 移植したコード: `src/torchaudio/functional/functional.py` の `spectrogram`、`melscale_fbanks` → `src/mel.rs`

```text
BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## BS-RoFormer

- URL: https://github.com/lucidrains/BS-RoFormer
- 対象: コミット `4b951924b57f46f817fc80ac17b828c7af01094d` で確認した `LICENSE`。`LICENSE` は `390cf293367cba1051f12b68daf549bd4b12e647` で追加された後、このコミットまで変更されていない。
- 移植したコード: Transformer の実装 → `src/model/roformer.rs`。Beat This! の `beat_this/model/roformer.py` の先頭に、BS-RoFormer に由来する旨が記載されている。

```text
MIT License

Copyright (c) 2023 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## rotary-embedding-torch

- URL: https://github.com/lucidrains/rotary-embedding-torch
- 対象: タグ `0.6.4`（コミット `75ad1fd7cb53c568cb4c8d6d56f04c2a6aae9623`）。ライセンスを確認した時点の Beat This! の `requirements.txt` で指定されているバージョンである。
- 移植したコード: `rotary_embedding_torch/rotary_embedding_torch.py` の回転位置埋め込み → `src/model/roformer.rs`

```text
MIT License

Copyright (c) 2021 Phil Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

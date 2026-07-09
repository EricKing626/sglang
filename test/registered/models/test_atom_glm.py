# =============================================================================
# ATOM + GLM-5.2 端到端 smoke test（sglang v0.5.12,ATOM 綁定版本）
# =============================================================================
# 這支 unittest 會「自己起一個 SGLang server」跑 GLM-5.2 + ATOM 後端,再跑一小段
# GSM8K 並斷言 accuracy——不用手動起 server / 發請求,一鍵驗證整條
# SGLang <-> ATOM <-> AITER 有沒有打通。
#
# 前置需求：
#   - ROCm GPU x8(TP=8);ATOM + AITER 已安裝且版本對齊(建議 rocm/atom-dev:latest 為底)
#   - HF_HOME 已設,GLM-5.2-FP8 可下載;或用 ATOM_GLM_MODEL 指向本地路徑
#
# 怎麼跑：
#   cd /sgl-workspace/sglang            # 這裡是 v0.5.12 的 checkout
#   export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH
#   # GLM-5.2（現行 ATOM main)的正確路徑 = external-model-package(非 --model-impl atom):
#   export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models
#   export ATOM_GLM_MODEL=zai-org/GLM-5.2-FP8     # 或 /data/models/GLM-5.2-FP8
#   python3 -m pytest -s test/registered/models/test_atom_glm.py
#   # -s 才會印出 metrics=
#
# 備註：
#   * 本檔預設走 external-model-package(ATOM 完整 wrapper,有 load_weights)。
#     若你的 ATOM 版本是「prepare_model_for_sglang 回傳完整模型」的舊版,可改成
#     在 other_args 加 ["--model-impl","atom"] 並移除下面那行 setdefault。
#   * 建議先跑 test_atom_models.py(Qwen、單卡、快)確認 adapter/環境,再跑本檔。
# =============================================================================

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# ATOM canonical integration: register ATOM's per-arch wrappers into sglang.
os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "atom.plugin.sglang.models")

# Override with a local path if desired: export ATOM_GLM_MODEL=/data/models/GLM-5.2-FP8
GLM_MODEL = os.environ.get("ATOM_GLM_MODEL", "zai-org/GLM-5.2-FP8")


class TestGLMAtom(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GLM_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size", "8",
                "--trust-remote-code",
                "--kv-cache-dtype", "fp8_e4m3",
                "--mem-fraction-static", "0.85",
                "--disable-radix-cache",
                "--tool-call-parser", "glm47",
                "--reasoning-parser", "glm45",
                # keep context modest for a smoke test; bump if you need long ISL
                "--context-length", "32768",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=64,          # small = fast smoke; raise to 200 for a real check
            max_new_tokens=512,
            parallel=64,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        # GLM-5.2 should easily clear this; lower it only if you shrink the model.
        self.assertGreater(metrics["accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()

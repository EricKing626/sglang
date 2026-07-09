# =============================================================================
# ATOM + GLM-5.2 端到端 smoke test —— `--model-impl atom` adapter 路徑
# =============================================================================
# 與 test_atom_glm.py 相同,但走 PR #16944 的 `--model-impl atom` adapter,而非
# external-model-package。此路徑呼叫 `atom.prepare_model_for_sglang(config)` 取得
# 模型,由 sglang 的 ATOMForCausalLM adapter 轉接 forward / load_weights。
#
# ⚠️ 適用版本:只有當你的 ATOM 版本的 `prepare_model_for_sglang` 回傳「具備
#    load_weights 的完整模型」時才可用(例如 PR 原綁的 zejun/plugin_for_atom_1223)。
#    對「回傳裸模型」的現行 ATOM main,請改用 test_atom_glm.py(external-package)。
#
# 怎麼跑:
#   cd /sgl-workspace/sglang            # v0.5.12 checkout
#   export PYTHONPATH=/sgl-workspace/sglang/python:$PYTHONPATH
#   export ATOM_GLM_MODEL=zai-org/GLM-5.2-FP8     # 或 /data/models/GLM-5.2-FP8
#   python3 -m pytest -s test/registered/models/test_atom_glm_modelimpl.py
#   # -s 才會印出 metrics=
#
# 備註:此檔「不」設 SGLANG_EXTERNAL_MODEL_PACKAGE,改在 other_args 傳
#      --model-impl atom(對應 sglang models/atom.py 的 ATOMForCausalLM)。
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

# Override with a local path if desired: export ATOM_GLM_MODEL=/data/models/GLM-5.2-FP8
GLM_MODEL = os.environ.get("ATOM_GLM_MODEL", "zai-org/GLM-5.2-FP8")


class TestGLMAtomModelImpl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GLM_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--model-impl", "atom",          # route model through ATOM adapter
                "--tp-size", "8",
                "--trust-remote-code",
                "--kv-cache-dtype", "fp8_e4m3",
                "--mem-fraction-static", "0.85",
                "--disable-radix-cache",
                "--tool-call-parser", "glm47",
                "--reasoning-parser", "glm45",
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
            num_questions=64,
            max_new_tokens=512,
            parallel=64,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.60)


if __name__ == "__main__":
    unittest.main()

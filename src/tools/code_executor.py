"""
安全 Python 代码执行器 - 沙箱化执行论文算法复现代码

特性:
- 子进程隔离 + 临时目录运行
- 超时保护 + 内存限制提示
- 导入白名单 + 危险模式阻止
- 沙箱 preamble 注入，禁止危险内置函数
- 结构化返回结果
"""

import subprocess
import tempfile
import sys
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


# ================================================================
#  沙箱配置
# ================================================================

# 危险导入黑名单（仅阻止网络/子进程调用，允许科学计算库内部依赖）
BLOCKED_IMPORTS = [
    "socket", "requests", "urllib", "urllib2", "urllib3",
    "http", "httplib", "http.client", "http.server",
    "ftplib", "smtplib", "telnetlib", "poplib", "imaplib",
    "subprocess", "shutil",
    "shelve",
    "multiprocessing",
    "pdb", "code", "codeop",
]

# 绝对禁止的模式（黑名单）
FORBIDDEN_PATTERNS = [
    "os.system", "os.popen", "os.remove", "os.rmdir", "os.unlink",
    "os.chdir", "os.exec", "os.spawn", "os.kill",
    "shutil.", "socket.", "requests.", "urllib.", "http",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "pickle.", "ctypes.", "multiprocessing.",
    "open(", "file(",
    "__import__",
    "eval(", "exec(",
]

# 注入到用户代码前的沙箱保护
SANDBOX_PREAMBLE = '''
# === 沙箱保护 (自动注入) ===
import builtins as __builtins__
import os as __os__
# 清除环境变量
for _k in list(__os__.environ.keys()):
    del __os__.environ[_k]
__os__.environ["PYTHONIOENCODING"] = "utf-8"
__os__.environ["PYTHONUTF8"] = "1"

_orig_import = __builtins__.__import__
_blocked = {blocked_imports}

def __safe_import__(name, *args, **kwargs):
    root = name.split(".")[0]
    if root in _blocked:
        raise ImportError(f"sandbox: import '{{name}}' blocked")
    return _orig_import(name, *args, **kwargs)

__builtins__.__import__ = __safe_import__
'''


# ================================================================
#  结果类型
# ================================================================

@dataclass
class ExecutionResult:
    """代码执行的完整结果"""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    elapsed_ms: int = 0
    error_type: str = ""       # "timeout" | "security" | "runtime" | ""
    error_message: str = ""
    truncated: bool = False


# ================================================================
#  沙箱执行器
# ================================================================

class SafeCodeExecutor:
    """安全的 Python 代码沙箱执行器"""

    def __init__(self, timeout: int = 30, max_output_chars: int = 5000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, code: str, input_data: str = "") -> ExecutionResult:
        """
        在沙箱中执行 Python 代码。

        Args:
            code: Python 源代码
            input_data: 可选的 stdin 输入

        Returns:
            ExecutionResult
        """
        # --- 静态安全检查 ---
        violations = self._security_check(code)
        if violations:
            return ExecutionResult(
                success=False,
                error_type="security",
                error_message=f"安全违规: {'; '.join(violations)}",
            )

        # --- 包装 + 注入保护 ---
        wrapped = self._wrap_code(code)

        # --- 写入临时文件并执行 ---
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_")
            os.close(fd)
            Path(tmp_path).write_text(wrapped, encoding="utf-8")

            t0 = time.time()
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
                env={
                    "PYTHONIOENCODING": "utf-8",
                    "PYTHONUTF8": "1",
                    "PATH": os.environ.get("PATH", ""),
                },
                input=input_data if input_data else None,
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            stdout = self._truncate(proc.stdout or "")
            stderr = self._sanitize_stderr(proc.stderr or "")

            return ExecutionResult(
                success=(proc.returncode == 0),
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
                elapsed_ms=elapsed_ms,
                truncated=(len(proc.stdout or "") > self.max_output_chars),
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error_type="timeout",
                error_message=f"执行超时 ({self.timeout}秒)",
                elapsed_ms=self.timeout * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_type="runtime",
                error_message=f"执行异常: {str(e)}",
            )
        finally:
            if tmp_path and Path(tmp_path).exists():
                try:
                    Path(tmp_path).unlink()
                except Exception:
                    pass

    # ================================================================
    #  Internal
    # ================================================================

    def _security_check(self, code: str) -> list[str]:
        """静态安全检查"""
        violations = []
        code_lower = code.lower()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.lower() in code_lower:
                violations.append(f"禁止操作: {pattern}")
        return violations

    def _wrap_code(self, code: str) -> str:
        """注入沙箱保护 preamble"""
        preamble = SANDBOX_PREAMBLE.format(
            blocked_imports=repr(set(BLOCKED_IMPORTS))
        )
        return f"{preamble}\n\n# === 用户代码 ===\n{code}"

    def _truncate(self, text: str) -> str:
        if len(text) > self.max_output_chars:
            return text[:self.max_output_chars] + "\n... [输出已截断]"
        return text

    def _sanitize_stderr(self, stderr: str) -> str:
        """清理 stderr 中泄露的文件路径"""
        stderr = re.sub(r'File ".*?sandbox_[^"]*"', 'File "<sandbox>"', stderr)
        stderr = re.sub(r'File ".*?Temp[^"]*sandbox[^"]*"', 'File "<sandbox>"', stderr)
        return stderr


# ================================================================
#  便捷函数
# ================================================================

def execute_code(code: str, timeout: int = 30) -> ExecutionResult:
    """快速安全执行 Python 代码"""
    executor = SafeCodeExecutor(timeout=timeout)
    return executor.execute(code)


def execute_and_format(code: str, timeout: int = 30) -> str:
    """
    执行代码并返回格式化的结果字符串。
    适配 Agent 工具返回值。
    """
    result = execute_code(code, timeout)

    if result.error_type == "security":
        return f"❌ 安全违规: {result.error_message}"

    if result.error_type == "timeout":
        return f"⏱️ 执行超时 ({timeout}秒)"

    parts = []
    if result.stdout:
        parts.append(f"📤 stdout:\n{result.stdout}")
    if result.stderr:
        parts.append(f"📤 stderr:\n{result.stderr}")
    if not result.stdout and not result.stderr:
        parts.append("✅ 代码执行完成，无输出。")

    if result.exit_code != 0:
        parts.append(f"⚠️ exit code: {result.exit_code}")
    else:
        parts.append(f"✅ 执行成功 ({result.elapsed_ms}ms)")

    return "\n\n".join(parts)

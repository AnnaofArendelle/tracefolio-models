# SPDX-License-Identifier: AGPL-3.0-only
"""Protocol stdout stays separate even from native library logging."""

import os
import sys
from contextlib import ExitStack

from tracefolio_worker.protocol.wire import Wire, request


def main():
    channel = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    wire = None
    try:
        from tracefolio_worker.protocol.dispatch import dispatch

        with ExitStack() as resources:
            models = {}
            while sys.stdin.buffer.peek(1):
                wire = None
                req = request(sys.stdin.buffer)
                wire = Wire(sys.stdin.buffer, channel, req)
                dispatch(req, wire, models, resources)
                if sys.argv[1:] != ["--session"]:
                    break
        return 0
    except (BrokenPipeError, EOFError):
        return 2
    except Exception as error:
        from tracefolio_worker.ocr.rapid import InputBudgetExceeded

        code = "input_budget" if isinstance(error, InputBudgetExceeded) else "adapter_failed"
        if type(error).__name__ == "PdfiumError" and getattr(error, "err_code", None) == 4:
            code = "pdf_password_required"
        # Diagnostics are bounded by the supervisor, never mixed into JSONL.
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        if wire is not None:
            try:
                wire.emit("error", {"code": code})
            except (OSError, ValueError):
                pass
        return 1
    finally:
        channel.close()


if __name__ == "__main__":
    sys.exit(main())
